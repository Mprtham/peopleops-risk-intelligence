with source as (
    select * from {{ source('peopleops_raw', 'employees') }}
)

select
    employee_id,
    full_name,
    gender,
    date_of_birth,
    department,
    hire_date,
    contract_type
from source
