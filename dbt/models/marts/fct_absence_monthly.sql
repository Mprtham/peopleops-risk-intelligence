select
    employee_id,
    date_trunc(absence_date, month)         as month,
    count(*)                                as absence_events,
    sum(days)                               as total_absence_days,
    countif(absence_type = 'Sick')          as sick_days,
    countif(absence_type = 'Personal')      as personal_days,
    countif(absence_type = 'Family')        as family_days,
    countif(absence_type = 'Medical')       as medical_days,
    countif(absence_type = 'Other')         as other_days
from {{ ref('stg_absence_events') }}
group by 1, 2
