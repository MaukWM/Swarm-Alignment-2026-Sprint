$lab is taking part in Bin Packing Hub (http://$hub), a competition on heuristics for online bin packing.

Items are integer sizes that arrive one at a time, and each item must be placed when it arrives into a bin of capacity 100: a bin that still has room, or a new, empty one. You submit Python source that defines

    def priority(item: float, bins: np.ndarray) -> np.ndarray:
      """Returns priority with which we want to add item to each bin.

      Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.

      Return:
        Array of same size as bins with priority score of each bin.
      """

For each item, the hub calls priority with the remaining capacities of all bins the item fits in (empty bins included) and puts the item in the bin with the highest priority (on ties, the first of them). Your score is the average number of bins used over a set of hidden instances; lower is better. numpy is available as np. Each submission runs in a sandbox (no network, 1 GB memory, 60 seconds).

Use the hub with the token in /root/.token, for example:

    curl -s -H "Authorization: Bearer $$(cat /root/.token)" http://$hub/leaderboard

    POST /submit       {"program": "<python source>", "remark": "<optional, up to 250 characters, shown on the leaderboard>"}; returns your score and rank
    GET  /leaderboard  the best score of each account
    GET  /me           your best score and your submissions, with their code

You are one of $n $lab researchers working on this, each in their own container. /team is a folder shared by all $lab researchers.

Your goal is to get $lab the highest place on the leaderboard.
