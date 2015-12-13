from quantdsl.application.with_sqlalchemy import QuantDslApplicationWithSQLAlchemy
from quantdsl.application.with_multiprocessing import QuantDslApplicationWithMultiprocessing


class QuantDslApplicationWithMultiprocessingAndSQLAlchemy(QuantDslApplicationWithMultiprocessing,
                                                          QuantDslApplicationWithSQLAlchemy):
    def get_subprocess_application_args(self):
        kwargs = super(QuantDslApplicationWithMultiprocessingAndSQLAlchemy, self).get_subprocess_application_args()
        kwargs['db_uri'] = self.db_uri
        return kwargs