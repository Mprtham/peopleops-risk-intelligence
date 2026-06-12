with source as (
    select * from {{ source('peopleops_raw', 'exits') }}
)

select
    exit_id,
    employee_id,
    exit_date,
    exit_reason
from source
