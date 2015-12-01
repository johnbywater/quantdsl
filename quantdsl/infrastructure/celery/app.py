from celery import Celery

celery_app = Celery('quantdsl.infrastructure.celery.tasks')
celery_app.config_from_object('quantdsl.infrastructure.celery.config')

