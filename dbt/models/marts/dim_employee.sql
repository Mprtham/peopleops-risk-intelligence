-- SCD Type 2 employee dimension.
-- One row per role/salary segment; valid_from/valid_to bracket the period.
-- is_current = true identifies the live record.

with employees as (
    select * from {{ ref('stg_employees') }}
),

role_history as (
    select * from {{ ref('stg_role_history') }}
)

select
    to_hex(md5(concat(
        cast(e.employee_id as string), '_',
        cast(rh.effective_from as string)
    ))) as employee_sk,

    e.employee_id,
    e.full_name,
    e.gender,
    e.date_of_birth,
    e.department,
    e.hire_date,
    e.contract_type,

    rh.job_role,
    rh.role_level,
    rh.salary,
    rh.effective_from as valid_from,
    coalesce(rh.effective_to, date('9999-12-31')) as valid_to,
    rh.is_current

from employees e
inner join role_history rh on e.employee_id = rh.employee_id
