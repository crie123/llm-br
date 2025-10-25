import inference

responder = inference.build_chat_responder(max_examples=500)
entries = getattr(responder, 'entries', [])
print('entries_count=', len(entries))
for i in range(min(5, len(entries))):
    e = entries[i]
    print('\n--- entry', i, 'keys=', list(e.keys()))
    snippet = inference._stringify_response(e.get('response'))
    print(' response_snippet=', snippet[:500].replace('\n', ' '))
