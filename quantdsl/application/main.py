import os

from quantdsl.application.with_cassandra import DEFAULT_QUANTDSL_CASSANDRA_KEYSPACE


def get_quantdsl_app(backend_name='pythonobjects', call_evaluation_queue=None):
    # Todo: Put this config stuff under test.
    backend = os.environ.get('QUANTDSL_BACKEND', backend_name).strip().lower()
    if backend == 'pythonobjects':
        from quantdsl.application.with_pythonobjects import QuantDslApplicationWithPythonObjects
        quantdsl_app = QuantDslApplicationWithPythonObjects(call_evaluation_queue=call_evaluation_queue)
    elif backend == 'sqlalchemy':
        from quantdsl.application.with_sqlalchemy import QuantDslApplicationWithSQLAlchemy
        db_uri = os.environ.get('QUANTDSL_DB_URI', 'sqlite:////tmp/quantdsl-tmp.db')
        quantdsl_app = QuantDslApplicationWithSQLAlchemy(db_uri=db_uri, call_evaluation_queue=call_evaluation_queue)
    elif backend == 'cassandra':
        from quantdsl.application.with_cassandra import QuantDslApplicationWithCassandra
        # hosts = [i.strip() for i in os.environ.get('QUANT_DSL_CASSANDRA_HOSTS', 'localhost').split(',')]
        keyspace = os.environ.get('QUANTDSL_CASSANDRA_KEYSPACE', DEFAULT_QUANTDSL_CASSANDRA_KEYSPACE).strip()
        # port = int(os.environ.get('QUANTDSL_CASSANDRA_PORT', '9042').strip())
        # username = os.environ.get('QUANTDSL_CASSANDRA_USERNAME', '').strip() or None
        # password = os.environ.get('QUANTDSL_CASSANDRA_PASSWORD', '').strip() or None
        quantdsl_app = QuantDslApplicationWithCassandra(
                default_keyspace=keyspace,
                # port=port,
                # username=username,
                # password=password,
                call_evaluation_queue=call_evaluation_queue
        )
    else:
        raise ValueError("Invalid backend ('sqlalchemy', 'cassandra' and 'pythonobjects' are supported): " + backend)
    return quantdsl_app