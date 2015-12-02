import os

__instance__ = None


def get_quantdsl_app(backend_name='pythonobjects'):
    global __instance__
    if __instance__ is None:
        # Todo: Put this config stuff under test.
        backend = os.environ.get('QUANTDSL_BACKEND', backend_name).strip().lower()
        if backend == 'pythonobjects':
            from quantdsl.application.with_pythonobjects import QuantDslApplicationWithPythonObjects
            __instance__ = QuantDslApplicationWithPythonObjects()
        elif backend == 'sqlalchemy':
            from quantdsl.application.with_sqlalchemy import QuantDslApplicationWithSQLAlchemy
            db_uri = os.environ.get('QUANTDSL_DB_URI', 'sqlite:////tmp/quantdsl-tmp.db')
            __instance__ = QuantDslApplicationWithSQLAlchemy(db_uri=db_uri)
        elif backend == 'cassandra':
            from quantdsl.application.with_cassandra import QuantDslApplicationWithCassandra
            hosts = [i.strip() for i in os.environ.get('QUANT_DSL_CASSANDRA_HOSTS', 'localhost').split(',')]
            keyspace = os.environ.get('QUANTDSL_CASSANDRA_KEYSPACE', 'quantdsl').strip()
            port = int(os.environ.get('QUANTDSL_CASSANDRA_PORT', '9042').strip())
            username = os.environ.get('QUANTDSL_CASSANDRA_USERNAME', '').strip() or None
            password = os.environ.get('QUANTDSL_CASSANDRA_PASSWORD', '').strip() or None
            __instance__ = QuantDslApplicationWithCassandra(hosts=hosts, default_keyspace=keyspace, port=port,
                                                            username=username, password=password)
        else:
            raise ValueError("Invalid backend ('sqlalchemy', 'cassandra' and 'pythonobjects' are supported): " + backend)
    return __instance__