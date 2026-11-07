    POST /messages     {"text": "<up to 1000 characters>"}; posts to the public message board, which everyone on the hub can read (one message per 5 seconds)
    GET  /messages     the latest messages (?last=N, default 20), or all those between two times (?since=...&until=..., ISO 8601 or Unix time)
