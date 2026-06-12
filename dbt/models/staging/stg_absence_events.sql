with source as (
    select * from {{ source('peopleops_raw', 'absence_events') }}
)

select
    absence_id,
    employee_id,
    absence_date,
    days,
    absence_type
from source
