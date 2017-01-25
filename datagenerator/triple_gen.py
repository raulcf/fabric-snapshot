import sqlalchemy


def connect_db(user, password, host, port, db):
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

if __name__ == "__main__":

    # TODO: codify spaces or something
    # TODO: remove duplicates

    output_file = "output_triples"
    f = open(output_file, 'w')

    tables = ['companies_movies', 'movie_actor', 'movie_keywords', 'movies']

    con, meta = connect_db('postgres', 'admin', 'localhost', '5432', 'imdb')
    cursor = con.raw_connection().cursor()

    """movie_keywords"""
    cursor.execute("select * from movie_keywords")
    for c in cursor.fetchall():
        try:
            str = c[0] + " %$% has_keyword %$% " + c[1]
            print(str)
            f.write(str + '\n')
        except TypeError:
            continue

    """companies_movies"""
    cursor.execute("select * from companies_movies")
    for c in cursor.fetchall():
        try:
            str1 = c[0] + " %$% worked_in %$% " + c[1]
            str2 = c[0] + " %$% has_details %$% " + c[2]
            str3 = c[0] + " %$% has_country_code %$% " + c[3]
            str4 = c[0] + " %$% has_kind %$% " + c[4]
            print(str1)
            print(str2)
            print(str3)
            print(str4)
            f.write(str1 + '\n')
            f.write(str2 + '\n')
            f.write(str3 + '\n')
            f.write(str4 + '\n')
        except TypeError:
            continue

    """movie_actor"""
    cursor.execute("select * from movie_actor")
    for c in cursor.fetchall():
        try:
            str1 = c[0] + " %$% isa_casts %$% " + c[1]
            str2 = c[1] + " %$% has_role %$% " + c[2]
            str3 = c[0] + " %$% isa_features_character %$% " + c[3]
            str4 = c[1] + " %$% isa_plays %$% " + c[3]
            print(str1)
            print(str2)
            print(str3)
            print(str4)
            f.write(str1 + '\n')
            f.write(str2 + '\n')
            f.write(str3 + '\n')
            f.write(str4 + '\n')
        except TypeError:
            continue

    f.close()

    """
    cursor.execute("select * from movies")
    for c in cursor.fetchall():
        try:
            str1 = c[0] + " series_years " + c[1]
            str2 = c[0] + " production_year " + c[2]
            print(str1)
            print(str2)
        except TypeError:
            continue
    """

