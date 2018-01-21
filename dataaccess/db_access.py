import sqlalchemy
from sqlalchemy.exc import ResourceClosedError
from sqlalchemy.sql.sqltypes import INTEGER, TEXT, VARCHAR


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
        return self.metadata.tables.keys()

    def list_attrs_of_table(self, table):
        """
        Returns attribute metadata of table
        :param table: the input table to describe
        :return: list of tuples with the metadata for each attribute
        """
        table_metadata = self.metadata.tables[table]
        columns = table_metadata.columns
        return [(c.name, c.type) for c in columns]

    def get_all_records(self, tables, read_batch_size=1000):
        cursor = self.con.raw_connection().cursor()
        for t in tables:
            cursor.execute("select * from " + str(t))
            while 1:
                try:
                    for c in cursor.fetchmany(size=read_batch_size):
                        yield c
                except ResourceClosedError:
                    print("done reading!")


def count_unique_text_records(conn, tables):
    unique_cells = set()
    total = len(tables)
    i = 0
    for t in tables:
        print(str(t) + ": " + str(i) + "/" + str(total))
        i += 1
        print("Total unique so far: " + str(len(unique_cells)))
        attributes_metadata = conn.list_attrs_of_table(t)
        proj_cols = [attributes_metadata[i][0] for i in range(len(attributes_metadata))
                   if type(attributes_metadata[i][1]) == VARCHAR or type(attributes_metadata[i][1]) == TEXT]
        cursor = conn.con.raw_connection().cursor()
        cursor.itersize = 20000
        for proj_col in proj_cols:
            cursor.execute("select " + str(proj_col) + " from " + str(t))
            while 1:
                result_set = cursor.fetchmany(size=10)
                if len(result_set) == 0:  # exit when finished
                    break
                for c in result_set:
                    h = hash(c)
                    unique_cells.add(h)
    print("Total unique cells: " + str(len(unique_cells)))


def _count_unique_text_records(conn, tables):
    unique_cells = set()
    for t in tables:
        print("Total unique so far: " + str(len(unique_cells)))
        attributes_metadata = conn.list_attrs_of_table(t)
        indexes = [i for i in range(len(attributes_metadata))
                   if type(attributes_metadata[i][1]) == VARCHAR or type(attributes_metadata[i][1]) == TEXT]
        cursor = conn.con.raw_connection().cursor()
        cursor.execute("select * from " + str(t))
        while 1:
            result_set = cursor.fetchmany(size=1000)
            if len(result_set) == 0:  # exit when finished
                break
            for c in result_set:
                for idx in indexes:
                    # print(c[idx])
                    unique_cells.add(c[idx])
            # except ResourceClosedError:
            #     print("done reading!")
    print("Total unique cells: " + str(len(unique_cells)))


if __name__ == "__main__":
    print("DB Access")

    conn = DBConn("postgres", "admin", "localhost", "5432", "drugcentral")

    print(conn.metadata)

    tables = conn.list_tables()
    # total_cols = 0
    # total_text_cols = 0
    # for t in tables:
    #     print("")
    #     print("###")
    #     print(str(t))
    #     column_metadata = conn.list_attrs_of_table(t)
    #     print(column_metadata)
    #     total_cols += len(column_metadata)
    #     for c_name, c_type in column_metadata:
    #         if type(c_type) == VARCHAR or type(c_type) == TEXT:
    #             total_text_cols += 1
    #     print("")
    #
    # print("Total columns: " + str(total_cols))
    # print("Total text columns: " + str(total_text_cols))

    # count_unique_text_records(conn, ['activities'])
    # exit()

    import time
    start = time.time()
    count_unique_text_records(conn, tables)
    stop = time.time()
    print("total: " + str(stop - start))


    # chembl_21   -> 14M
    # imdb        -> 36M
    # drugcentral ->  1M


