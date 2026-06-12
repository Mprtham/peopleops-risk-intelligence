-- Custom data test: returns rows when any review or absence event
-- post-dates the employee's exit. Zero rows = test passes.

with review_violations as (
    select
        r.employee_id,
        r.review_date as event_date,
        x.exit_date,
        'performance_review' as event_type
    from {{ ref('stg_performance_reviews') }} r
    inner join {{ ref('stg_exits') }} x on r.employee_id = x.employee_id
    where r.review_date > x.exit_date
),

absence_violations as (
    select
        a.employee_id,
        a.absence_date as event_date,
        x.exit_date,
        'absence_event' as event_type
    from {{ ref('stg_absence_events') }} a
    inner join {{ ref('stg_exits') }} x on a.employee_id = x.employee_id
    where a.absence_date > x.exit_date
)

select * from review_violations
union all
select * from absence_violations
