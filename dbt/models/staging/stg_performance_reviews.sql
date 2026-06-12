with source as (
    select * from {{ source('peopleops_raw', 'performance_reviews') }}
)

select
    review_id,
    employee_id,
    review_date,
    rating
from source
