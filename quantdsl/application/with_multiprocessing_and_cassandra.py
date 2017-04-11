from quantdsl.application.with_cassandra import QuantDslApplicationWithCassandra
from quantdsl.application.with_multiprocessing import QuantDslApplicationWithMultiprocessing


class QuantDslApplicationWithMultiprocessingAndCassandra(QuantDslApplicationWithMultiprocessing,
                                                         QuantDslApplicationWithCassandra):

    pass

