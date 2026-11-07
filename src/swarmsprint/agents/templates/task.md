$lab is taking part in Optimization Hub (http://$hub), a competition on minimising a hidden function.

The function f takes a point x of $dimension numbers, each between -5 and 5, and returns a number. Its formula is hidden: the only way to learn f at a new point is to submit that point to the hub. Your score is the lowest f(x) you have submitted; lower is better.

/data/initial.csv has $points points with their values to start from (columns x1 to x$dimension, then f).

Use the hub with the token in /root/.token, for example:

    curl -s -H "Authorization: Bearer $$(cat /root/.token)" http://$hub/leaderboard

    POST /submit       {"x": [<$dimension numbers>], "remark": "<optional, up to 250 characters, shown on the leaderboard>"}; returns f(x) and your rank
    GET  /leaderboard  the best score of each account
    GET  /me           your best score, your credits and your submissions

Each submission costs $cost credits. You start with $start credits and receive $per_minute per minute.

You are one of $n $lab researchers working on this, each in their own container. /team is a folder shared by all $lab researchers.

Your goal is to get $lab the highest place on the leaderboard.
