## Domain model objects.

class DomainObject(object): pass


class CallRequirement(DomainObject):
    def __init__(self, id, stubbed_expr_str, effective_present_time, required_call_ids, notify_ids):
        self.id = id
        self.stubbed_expr_str = stubbed_expr_str
        self.effective_present_time = effective_present_time
        self.required_call_ids = required_call_ids
        self.notify_ids = notify_ids
        # Todo: Validate.


class Result(DomainObject):
    def __init__(self, id, return_value):
        self.id = id
        self.value = return_value
