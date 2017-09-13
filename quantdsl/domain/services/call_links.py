def regenerate_execution_order(contract_specification_id, call_link_repo):
    # assert isinstance(call_link_repo, CallLinkRepository)
    link_id = contract_specification_id
    while True:
        call_id = get_next_call_id(call_link_repo, link_id)
        if call_id is None:
            break
        yield call_id
        if call_id == contract_specification_id:
            break
        link_id = call_id


def get_next_call_id(call_link_repo, link_id):
    try:
        call_link = call_link_repo[link_id]
    except KeyError:
        pass
    else:
        # assert isinstance(call_link, CallLink)
        return call_link.call_id
