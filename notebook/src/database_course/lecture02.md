# Lecture 2: Modern SQL

1971, the first relational query language called SQUARE was created. 1972 SEQUEL (Structured English Query Language) was created. SQL was added to ANSI standard in 1986. Current standard is SQL:2023.
- SQL:2023 - property graph queries, multi-dim arrays
- SQL:2016 - JSON, polymorphic tables etc.

Relational languages:
- Data Manipulation Langauge (DML)
- Data Definition Language (DDL)
- Data Control Language (DCL)

Important: <SQL is based on bags> (with duplicates) not sets (no duplicates).

We should try to do everything on the database, in one big query.

Example database:
- student table: `sid`, `login`, `gpa` etc.
- enrolled table: `sid`, `cid` (course id), `grade`
- course table: `cid`, `name`

Aggregates:
- `AVG(col)`. e.g. `SELECT AVG(s.gpa) FROM student as s`
- `MIN(col)`
- `MAX(col)`
- `COUNT(col)`. e.g. `SELECT COUNT(LOGIN) as cnt FROM student WHERE login LIKE '%@cs`. Equivalently, `COUNT(1)`. Count number of rows where their login matches the pattern.

Groupby: Get average gpa by course id.
```sql
SELECT AVG(s.gpa), e.cid
    FROM enrolled as e JOIN student AS s
    ON e.sid = s.sid
GROUP BY e.cid
```

String operations. 
- `LIKE` is used for string matching. `%` matches any substring, including empty strings. `_` matches any one character.
- SQL-92 defines string functions. 

Window functions. Perform a sliding calculation across a set of tuples that are related. Like an aggregation but tuples are not grouped into a single output tuple.
```sql
SELECT ... FUNC-NAME(...) OVER (...)
    FROM TABLE_NAME
```
e.g. Get row number per course id.
```sql
SELECT cid, sid
    ROW_NUMBER() OVER (PARTITION BY cid)
    FROM enrolled
ORDER BY cid
```

Nested queries. Invoke a query inside of another query to compose more complex computations. These are often difficult for the DBMS to optimize. 
e.g. This one below is a join written as a nested query. 
```sql
outer query ->    SELECT name FROM student WHERE
                    sid IN (SELECT sid FROM enrolled) <- inner query
```

e.g. Get the names of students in '15-445':
```sql
SELECT name FROM student
    WHERE sid IN (
        SELECT sid FROM enrolled
        WHERE cid = '15-445'
    )
```

Lateral joins. `LATERAL` operator allows us to reference tables preceding the expression. e.g. below, the second expression can reference t1.
```sql
SELECT * FROM (SELECT 1 AS X) AS t1,
    LATERAL (SELECT t1.x+1 AS y) AS t2;
```

Common Table Expressions. Provides a way to write auxiliary statements for use in a larger query, i.e. a temp table just for one query.
```sql
WITH cteName (col1, col2) as (
    SELECT 1, 2
)
SELECT col1 + col2 FROM cteName
```

Demonstration of CTEs: Find student record with the highest id that is enrolled in at least one course. We use the maxId in the temporary table below.
```sql
WITH cteCourse (maxId) AS (
    SELECT MAX(sid) FROM enrolled
)
SELECT name FROM student, cteSource
    WHERE student.sid = cteSource.maxId
```

We can also use recursion with CTE. Print the sequence of numbers from 1 to 10.
```sql
WITH RECURSIVE cteSource (counter) AS (
    (SELECT 1)
    UNION
    (SELECT counter + 1 FROM cteSource
    WHERE counter < 10)
)
SELECT * FROM cteSource
```



