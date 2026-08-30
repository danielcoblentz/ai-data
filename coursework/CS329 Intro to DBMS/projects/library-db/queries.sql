-- library-db/queries.sql
SELECT * FROM books;
SELECT * FROM members;
SELECT b.title, m.name FROM loans l JOIN books b ON l.book_id = b.id JOIN members m ON l.member_id = m.id;
