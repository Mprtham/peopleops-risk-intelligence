with source as (
    select * from {{ source('peopleops_raw', 'role_history') }}
)

select
    employee_id,
    job_role,
    role_level,
    salary,
    effective_from,
    effective_to,
    effective_to is null as is_current
from source
