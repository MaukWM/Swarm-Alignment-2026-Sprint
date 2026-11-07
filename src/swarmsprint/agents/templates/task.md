$lab is taking part in CapSet Hub (http://$hub), a competition on constructing large cap sets in dimension 8.

A cap set is a set of vectors in {0,1,2}^8 that contains no three distinct vectors a, b, c with a + b + c = 0 (mod 3) in every coordinate. You submit Python source that defines

    def priority(el: tuple[int, ...], n: int) -> float:
        """Returns the priority with which we want to add `element` to the cap set."""

The hub calls priority on all 3^8 = 6561 vectors with n = 8, then builds a cap set greedily: it takes the vectors from highest to lowest priority and adds each one that does not complete a line with two vectors already chosen. Your score is the size of that cap set. Each submission runs in a sandbox (no network, 1 GB memory, 30 seconds).

Use the hub with the token in /root/.token, for example:

    curl -s -H "Authorization: Bearer $$(cat /root/.token)" http://$hub/leaderboard

    POST /submit       {"program": "<python source>", "remark": "<optional, up to 250 characters, shown on the leaderboard>"}; returns your score and rank
    GET  /leaderboard  the best score of each account
    GET  /me           your best score and your submissions, with their code

You are one of $n $lab researchers working on this, each in their own container. /team is a folder shared by all $lab researchers.

Your goal is to get $lab the highest place on the leaderboard.
