import sqlalchemy


class DBConn:

    def __init__(self, user, password, host, port, db):
        con, meta = self.connect_db(user, password, host, port, db)
        self.con = con
        self.metadata = meta

    def connect_db(self, user, password, host, port, db):
        """
        Connects to postgresql db and returns connection and db object
        :param user:
        :param password:
        :param host:
        :param port:
        :param db:
        :return:
        """
        url = 'postgresql://{}:{}@{}:{}/{}'
        url = url.format(user, password, host, port, db)
        con = sqlalchemy.create_engine(url, client_encoding='utf8')
        meta = sqlalchemy.MetaData(bind=con, reflect=True)
        return con, meta

    def release_db(self):
        self.con.close()

    def list_tables(self):
        """
        Returns list of tables in the database
        :return:
        """
        return self.metadata.tables().keys()

    def get_all_records(self, tables):
        cursor = self.con.raw_connection().cursor()
        for t in tables:
            cursor.execute("select * from " + str(t))
            for c in cursor.fetchall():
                yield c

if __name__ == "__main__":
    print("DB Access")

    conn = DBConn("postgres", "admin", "localhost", "5432", "imdb")

    print(conn.metadata)

    tables = conn.list_tables()
    for t in tables:
        print(str(t))
