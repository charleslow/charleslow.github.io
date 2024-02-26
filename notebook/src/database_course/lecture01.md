# Lecture 1

Course about how to design and implement a database management system. Textbook: <Database System Concepts> by Silberschatz, Korth and Sudarshan. 

Agenda:
- Database Systems Background
- Relational Model
- Relational Algebra
- Alternative Data Models

A database is an organized collection of inter-related data that models some aspect of the real world. Databases are the core component of most computer applications. e.g. an excel spreadsheet is a database. SQLite is the most popular database as it is used in every cellphone. 

## Flat File Strawman

Store our database as a csv file that we manage. e.g. each line corresponds to an artist, year, country etc. Problems:
- Super slow to find the artist of interest as we use a for-loop to find each
- Super slow to update or delete an artist
- Data types are not stored on the csv file, we need to know which is an integer etc.
- Concurrent writes to the file are not supported

A <Database Management System (DBMS)> is a software that allows applications to store and analyze information in a database. A general purpose DBMS supports the definition, creation, querying, update and administrations of databases in accordance with some data model. Usually first choice is postgres or sqlite.

A <data model> is a collection of concepts for describing the data in a database. A <schema> is a description of a particular collection of data, using a given data model. Examples of data models:
1. Relational
2. Key / Value 
3. Graph
4. Document / XML / Object
5. Wide-Column / Column-family
6. Array / Matrix / Vectors
7. Hierarchical
8. Network
9. Multi-Value

1 is the most common. 2-4 are considered NoSQL models (a loose term). 7-9 are obsolete.

## Relational Model

Early database applications were difficult to write. Every time the database schema or layout changed, IBM would need to rewrite database programs. Ted Codd devised the relational model to address this problem. The relational model is an abstraction. The relational model defines a database abstraction based on relations to reduce maintenance overhead. Key tenets:
- Store database in simple data structures (relations)
- Physical storage left up to the DBMS implementation
- Access data through high-level language, DBMS figures out the best execution model

A <relation> is an unordered set that contain the relationship of attributes that represent entities. An n-ary relation is equivalent to a table with n columns. A <tuple> is a set of attributes values (also known as its domain) in the relation. The special value `NULL` is a member of every domain.

A relation's <primary key> uniquely identifies a single tuple. Some DBMSs automatically create an internal primary key if a table does not define one. Primary key is a constraint that the DBMS will enforce to ensure no duplicates exist. A <foreign key> specifies that an attribute from one relation maps to a tuple in another relation. E.g. If we have an artist table with the artist id, and an album table with an artist column, the artist column is a foreign key.

We can impose <constraints> on the database that must hold for any tuple. DBMS will then prevent any modification that could violate those constraints. Unique and foreign key constraints are the most common. e.g. `CREATE ASSERTION` in SQL. 

## Data Manipulation Languages (DML)

There are two broad methods to store and retrieve information from a database:
1. Procedural: the query specifies the high level strategy to find the desired result based on sets. This uses relational algebra.
2. Non-Procedural (Declarative): The query specifies only what data is wanted and not how to find it. This uses relational calculus.

## Relational Algebra

Fundamental operations to retrieve and manipulate tuples in a relation. Based on set algebra (unordered lists with no duplicates). Each operator takes one or more relations as its inputs and outputs a new relation. We can thus chain operators together to create more complex operations. The operations are:

- <SELECT>. Choose a subset of the tuples from a relation that satisfies a selection predicate (filter). Predicates act as filters to retain only tuples that fulfill the qualifying requirement. We can combine multiple predicates using conjunctions / disjunctions.
    - Syntax: $\sigma_{predicate}(R)$
    - `SELECT * from TABLE where id="a"`
- <PROJECTION>. Generate a relation with tuples that contains only the specified attributes. E.g. re-arrange ordering, manipulate values ($+,-$ etc.) and remove unwanted attributes.
    - Syntax: $\pi_{A1, A2, ...}(R)$
    - Example: `SELECT b_{id} - 100`
- <UNION>. Generate a relation that contains all tuples that appear in one or both input relations. Note that R and S must have the same schema.
    - Syntax: $(R \bigcup S)$
    - Example: `(SELECT * from R) UNION (SELECT * from S)`
- <INTERSECTION>. Generate a relation that contains only the tuples that appear in both of the input relations.
    - Syntax: $(R \bigcap S)$
    - Example: `(SELECT * from R) INTERSECT (SELECT * from S)`
- <DIFFERENCE>. Generate a relation that contains only the tuples that appear in the first and not the second of the input relations.
    - Syntax: $(R - S)$
    - Example: `(SELECT * from R) EXCEPT (SELECT * from S)`
- <JOIN>. Generate a relation that contains all tuples that are a combination of two tuples (one from each input relation) with a common value for one or more attributes.
    - Syntax: $(R \ \infty \ S)$
    - Example: `SELECT * FROM R NATURAL JOIN S;`
- <Rename> $\rho$
- <Assignment> $R \leftarrow S$
- <Duplicate Elimination> $\delta$
- <Aggregation> $\gamma$
- <Sorting> $\tau$
- <Division> $(R \div S)$

Relational algebra defines an ordering of the high level steps of how to compute a query. E.g.
- $\sigma_{b_{id}=102}(R \ \infty \ S)$ vs $R \ \infty \ (\sigma_{b_{id}=102}(S))$. The former will do a huge join before filtering, whereas the latter filters first before joining, which is much better.

Instead of specifying the exact operations, DBMS allow us to state the high level answer we want, and the DBMS decides on the exact operations and the ordering to perform. This abstracts away the need for the user to know which operations are more efficient. Note that the relational model is independent of the query language implementation, although SQL is the de-facto standard (with many variants).

## Document Data Model

A collections of record documents containing a hierarchy of named field / value pairs. A field's value can be a scalar type, array, or another document. Modern implementations use `JSON`. Main reason for this model is to avoid <relational object impedance msimatch>, i.e. relational databases store data in rows with relationships between tables, but in object oriented languages like Python data is stored in objects with nested attributes, which could result in inefficient queries when we try to map between the two. In contrast, Document Databases store data in a nested json which closely resembles the object-oriented approach, making it easier to work with. The down side is that we could end up storing a lot of duplicate data in the json objects.
- Examples: MongoDB, RavenDB, DynamoDB etc.

## Vector Data Model

One-dimensional arrays used for nearest neighbour search, used for semantic search on embeddings generated by transformer models. Native integration with modern ML tools etc. At their core, these are just systems that use specialized indexes (e.g. Meta FAISS) to perform NN search quickly.
- Examples: Pinecone, Weaviate, Milvus etc.