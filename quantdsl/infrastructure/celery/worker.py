from quantdsl.infrastructure.celery.tasks import celery_app, get_app_for_celery_task, celery_evaluate_call

quantdsl_app = get_app_for_celery_task()

