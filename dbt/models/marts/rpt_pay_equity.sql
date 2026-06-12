-- Pay equity report: median salary by gender × department × role level.
-- pay_gap_pct is the gap relative to the overall (all-gender) median
-- for that department × role band — positive means above median.
-- Strong talking point: shows structural pay gaps, not just raw averages.

with current_employees as (
    select
        e.employee_id,
        e.gender,
        e.department,
        rh.job_role,
        rh.role_level,
        rh.salary
    from {{ ref('stg_employees') }} e
    inner join {{ ref('stg_role_history') }} rh
        on  e.employee_id = rh.employee_id
        and rh.is_current
),

by_gender as (
    select
        department,
        role_level,
        job_role,
        gender,
        approx_quantiles(salary, 2)[offset(1)] as median_salary,
        avg(salary)                             as avg_salary,
        count(*)                                as headcount
    from current_employees
    group by 1, 2, 3, 4
),

overall as (
    select
        department,
        role_level,
        approx_quantiles(salary, 2)[offset(1)] as overall_median_salary
    from current_employees
    group by 1, 2
)

select
    b.department,
    b.role_level,
    b.job_role,
    b.gender,
    b.median_salary,
    b.avg_salary,
    b.headcount,
    o.overall_median_salary,
    safe_divide(
        b.median_salary - o.overall_median_salary,
        o.overall_median_salary
    ) as pay_gap_pct
from by_gender b
inner join overall o
    on  b.department = o.department
    and b.role_level = o.role_level
order by b.department, b.role_level, b.gender
