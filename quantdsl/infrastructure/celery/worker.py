from quantdsl.infrastructure.celery.tasks import get_quant_dsl_app_for_celery_worker, celery_app

# Celery looks for a celery application instance.
app = celery_app

# Initialise the application at the start, useful for threaded celery workers such as '-P eventlet'.
quantdsl_app = get_quant_dsl_app_for_celery_worker()