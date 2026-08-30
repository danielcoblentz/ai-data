-- library-db/sample_data.sql
INSERT INTO books (title, author, isbn) VALUES ('The Hobbit', 'J.R.R. Tolkien', '978-0261102217');
INSERT INTO members (name, email) VALUES ('Alice', 'alice@library.com');
INSERT INTO loans (book_id, member_id) VALUES (1, 1);
