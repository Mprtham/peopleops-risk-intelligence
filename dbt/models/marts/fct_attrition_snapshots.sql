-- Leakage-proof attrition feature table.
-- One row per employee per quarter-end snapshot date.
-- All features are computed strictly AS OF the snapshot date.
-- Label exited_within_6m looks forward from snapshot — this is intentional
-- and not leakage because the label is the target, not a feature.
-- Employees already exited at snapshot time are excluded.

with snapshots as (
    -- Quarter-end dates covering the observation window
    select snapshot_date
    from unnest([
        date '2023-03-31', date '2023-06-30', date '2023-09-30', date '2023-12-31',
        date '2024-03-31', date '2024-06-30', date '2024-09-30', date '2024-12-31',
        date '2025-03-31', date '2025-06-30', date '2025-09-30', date '2025-12-31'
    ]) as snapshot_date
),

employees as (select * from {{ ref('stg_employees') }}),
role_history as (select * from {{ ref('stg_role_history') }}),
reviews as (select * from {{ ref('stg_performance_reviews') }}),
absences as (select * from {{ ref('stg_absence_events') }}),
exits as (select * from {{ ref('stg_exits') }}),

active_at_snapshot as (
    -- Employees hired on/before snapshot and not yet exited at snapshot
    select
        e.employee_id,
        s.snapshot_date
    from employees e
    cross join snapshots s
    left join exits x on e.employee_id = x.employee_id
    where
        e.hire_date <= s.snapshot_date
        and (x.exit_date is null or x.exit_date > s.snapshot_date)
),

role_at_snapshot as (
    -- The one role segment that was active at the snapshot date
    select
        a.employee_id,
        a.snapshot_date,
        rh.job_role,
        rh.role_level,
        rh.salary,
        rh.effective_from as role_start_date
    from active_at_snapshot a
    inner join role_history rh
        on  a.employee_id = rh.employee_id
        and rh.effective_from <= a.snapshot_date
        and (rh.effective_to is null or rh.effective_to >= a.snapshot_date)
),

role_medians as (
    -- Salary medians by department × role_level (computed once from current segments)
    select
        e.department,
        rh.role_level,
        approx_quantiles(rh.salary, 2)[offset(1)] as median_salary
    from role_history rh
    inner join employees e on rh.employee_id = e.employee_id
    where rh.is_current
    group by 1, 2
),

avg_ratings as (
    -- Average of the most recent 2 reviews and average of ALL reviews before snapshot
    select
        employee_id,
        snapshot_date,
        avg(case when rn <= 2 then rating end) as avg_rating_last2,
        avg(rating)                             as avg_rating_all,
        count(*)                                as n_ratings
    from (
        select
            a.employee_id,
            a.snapshot_date,
            r.rating,
            row_number() over (
                partition by a.employee_id, a.snapshot_date
                order by r.review_date desc
            ) as rn
        from active_at_snapshot a
        inner join reviews r
            on  a.employee_id = r.employee_id
            and r.review_date <= a.snapshot_date
    )
    group by 1, 2
),

absence_last90 as (
    select
        a.employee_id,
        a.snapshot_date,
        coalesce(sum(ab.days), 0) as absence_days_last90,
        coalesce(count(ab.absence_id), 0) as absence_events_last90
    from active_at_snapshot a
    left join absences ab
        on  a.employee_id = ab.employee_id
        and ab.absence_date between date_sub(a.snapshot_date, interval 90 day)
                                and a.snapshot_date
    group by 1, 2
),

absence_lifetime as (
    -- All absences from hire to snapshot — normalized by tenure to remove length bias
    select
        a.employee_id,
        a.snapshot_date,
        coalesce(count(ab.absence_id), 0)  as absence_events_lifetime,
        safe_divide(
            coalesce(count(ab.absence_id), 0),
            nullif(date_diff(a.snapshot_date, max(e.hire_date), month), 0)
        )                                  as absence_events_per_month
    from active_at_snapshot a
    inner join employees e on a.employee_id = e.employee_id
    left join absences ab
        on  a.employee_id = ab.employee_id
        and ab.absence_date <= a.snapshot_date
    group by 1, 2
),

exit_label as (
    select
        a.employee_id,
        a.snapshot_date,
        case
            when x.exit_date between a.snapshot_date
                                 and date_add(a.snapshot_date, interval 6 month)
            then 1 else 0
        end as exited_within_6m,
        x.exit_date
    from active_at_snapshot a
    left join exits x on a.employee_id = x.employee_id
)

select
    -- Keys
    a.employee_id,
    a.snapshot_date,

    -- Demographics
    e.gender,
    e.department,
    e.contract_type,

    -- Time-based features (as of snapshot)
    date_diff(a.snapshot_date, e.date_of_birth, year)  as age_at_snapshot,
    date_diff(a.snapshot_date, e.hire_date, year)      as tenure_years,
    date_diff(a.snapshot_date, e.hire_date, month)     as tenure_months,

    -- Role & compensation (as of snapshot)
    r.job_role,
    r.role_level,
    r.salary,
    coalesce(rm.median_salary, r.salary)               as role_median_salary,
    safe_divide(
        r.salary - coalesce(rm.median_salary, r.salary),
        coalesce(rm.median_salary, r.salary)
    )                                                  as salary_vs_median_pct,
    date_diff(a.snapshot_date, r.role_start_date, month) as months_in_current_role,

    -- Performance (as of snapshot)
    coalesce(ar.avg_rating_last2, 3.0)                 as avg_rating_last2,
    coalesce(ar.avg_rating_all,   3.0)                 as avg_rating_all,
    coalesce(ar.n_ratings, 0)                          as n_ratings,

    -- Absence (trailing 90 days)
    ab.absence_days_last90,
    ab.absence_events_last90,

    -- Absence (lifetime rate — stronger signal, less window noise)
    coalesce(al.absence_events_per_month, 0.0)         as absence_events_per_month,

    -- Label (forward-looking — not a feature)
    el.exited_within_6m

from active_at_snapshot a
inner join employees e          on a.employee_id = e.employee_id
inner join role_at_snapshot r   on a.employee_id = r.employee_id
                               and a.snapshot_date = r.snapshot_date
left join  role_medians rm      on e.department = rm.department
                               and r.role_level = rm.role_level
left join  avg_ratings ar       on a.employee_id = ar.employee_id
                               and a.snapshot_date = ar.snapshot_date
left join  absence_last90 ab    on a.employee_id = ab.employee_id
                               and a.snapshot_date = ab.snapshot_date
left join  absence_lifetime al  on a.employee_id = al.employee_id
                               and a.snapshot_date = al.snapshot_date
inner join exit_label el        on a.employee_id = el.employee_id
                               and a.snapshot_date = el.snapshot_date
