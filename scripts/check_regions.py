from dotenv import load_dotenv
load_dotenv()

from src.utils.db import fetch_all

cols, rows = fetch_all("""
    select region,
           count(*) as n,
           min(time) as lo,
           max(time) as hi,
           count(distinct extract(year from time)) as yrs
    from measurements
    group by region
    order by n desc
""")

print(cols)
for r in rows:
    print(r)