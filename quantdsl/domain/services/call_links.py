def regenerate_execution_order(contract_specification_id, call_link_repo):
    # assert isinstance(call_link_repo, CallLinkRepository)
    call_id = contract_specification_id
    while True:
        call_id = call_link_repo[call_id].call_id
        yield call_id
        if call_id == contract_specification_id:
            break
