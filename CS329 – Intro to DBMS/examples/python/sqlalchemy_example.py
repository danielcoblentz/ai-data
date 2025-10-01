"""sqlalchemy_example.py
Minimal SQLAlchemy example (declarative) using SQLite.
"""
from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, create_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from datetime import datetime
from pathlib import Path

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    email = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    posts = relationship('Post', back_populates='user')

class Post(Base):
    __tablename__ = 'posts'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    title = Column(String, nullable=False)
    body = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship('User', back_populates='posts')

BASE_DIR = Path(__file__).resolve().parent
DB_FILE = BASE_DIR / '..' / 'db' / 'example_sa.db'
DB_FILE.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(f'sqlite:///{DB_FILE}', echo=False, future=True)
Session = sessionmaker(bind=engine)

# create tables
Base.metadata.create_all(engine)

# insert sample data (idempotent guard)
session = Session()
if session.query(User).count() == 0:
    alice = User(username='alice', email='alice@example.com')
    bob = User(username='bob', email='bob@example.com')
    carol = User(username='carol', email='carol@example.com')
    session.add_all([alice, bob, carol])
    session.commit()

    session.add_all([
        Post(user_id=alice.id, title='Hello world', body='First post by Alice'),
        Post(user_id=alice.id, title='Another post', body='Alice again'),
        Post(user_id=bob.id, title='Bob\'s post', body='Post by Bob'),
    ])
    session.commit()

print('SQLAlchemy DB created at', DB_FILE)
for u in session.query(User).all():
    print(u.id, u.username, 'posts=', len(u.posts))

session.close()
