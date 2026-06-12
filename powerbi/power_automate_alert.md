# Power Automate: Monthly Attrition KPI Alert

## Trigger
Scheduled — 1st of each month, 08:00.

## Flow steps

1. **Run a BigQuery query** (via the BigQuery connector or HTTP action)
   ```sql
   select
       count(*) as exits_this_month
   from `YOUR_PROJECT.peopleops_prod.fct_attrition_snapshots`
   where
       snapshot_date = (select max(snapshot_date) from `YOUR_PROJECT.peopleops_prod.fct_attrition_snapshots`)
       and exited_within_6m = 1
   ```

2. **Get active headcount** from the same snapshot.

3. **Calculate attrition rate** = exits / headcount.

4. **Condition**: If attrition rate > 0.05 (5%)

   - **Yes → Send email / Teams message** to HR leadership:
     ```
     Subject: ⚠ PeopleOps Alert — Monthly Attrition Above Threshold
     Body: Attrition rate for [month] is [rate]%, exceeding the 5% threshold.
           [link to Power BI dashboard]
           Recommended action: review high-risk employee list in the dashboard.
     ```

   - **No → End** (no notification)

## Setup notes
- Use the **Power BI** connector → **Refresh a dataset** action as an alternative
  trigger (fires after each dataset refresh rather than on a schedule).
- Store credentials in Power Automate environment variables — never hardcode them.
- Test the flow with a temporary threshold of 0% to confirm notifications land.
