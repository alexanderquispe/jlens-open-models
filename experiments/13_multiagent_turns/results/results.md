Calls replayed: 176 (141 team, 35 coordinator), from 4 episodes. Blocks: 64.

## 1. Replay check

Rebuilt prompts with the token count the server logged: 100.0%.

| position | mean probability of the written token | written token is the model's top token |
|---|---|---|
| A | 0.4 | 0.506 |
| B | 0.535 | 0.665 |
| C | 0.932 | 0.955 |

## 2. Next tool

Calls that go on to call a tool: 168. Always guessing the most common first tool (dashboard) is right 27.4% of the time.

| lens | position | third | median rank | in top 10 | right tool of 14 |
|---|---|---|---|---|---|
| J-lens | B | early | 79814.0 | 0.0 | 0.103 |
| J-lens | B | middle | 37975.5 | 0.0 | 0.161 |
| J-lens | B | late | 18231.5 | 0.0 | 0.144 |
| J-lens | C | early | 100458.0 | 0.0 | 0.13 |
| J-lens | C | middle | 15152.0 | 0.0 | 0.291 |
| J-lens | C | late | 4063.25 | 0.005 | 0.316 |
| logit lens | B | early | 6840.5 | 0.0 | 0.004 |
| logit lens | B | middle | 9093.0 | 0.0 | 0.042 |
| logit lens | B | late | 18241.5 | 0.001 | 0.098 |
| logit lens | C | early | 5700.5 | 0.0 | 0.006 |
| logit lens | C | middle | 4983.5 | 0.001 | 0.093 |
| logit lens | C | late | 8223.75 | 0.006 | 0.178 |

## 3. Act or not

AUC of log P(act words) - log P(wait words) for calls that change jobs against the other calls, averaged over thirds of the depth, with 95% intervals from resampling calls. 0.5 is chance.
Reference without activations (the agent's previous call was an observation): AUC 0.660.

| calls | n change | n other | lens | position | third | auc | low | high |
|---|---|---|---|---|---|---|---|---|
| all | 32 | 144 | J-lens | A | early | 0.446 | 0.337 | 0.553 |
| all | 32 | 144 | J-lens | A | middle | 0.44 | 0.327 | 0.553 |
| all | 32 | 144 | J-lens | A | late | 0.517 | 0.42 | 0.615 |
| all | 32 | 144 | J-lens | B | early | 0.648 | 0.551 | 0.742 |
| all | 32 | 144 | J-lens | B | middle | 0.707 | 0.608 | 0.797 |
| all | 32 | 144 | J-lens | B | late | 0.598 | 0.489 | 0.701 |
| all | 32 | 144 | J-lens | C | early | 0.798 | 0.701 | 0.883 |
| all | 32 | 144 | J-lens | C | middle | 0.796 | 0.702 | 0.881 |
| all | 32 | 144 | J-lens | C | late | 0.853 | 0.768 | 0.927 |
| all | 32 | 144 | logit lens | A | early | 0.536 | 0.419 | 0.659 |
| all | 32 | 144 | logit lens | A | middle | 0.483 | 0.377 | 0.589 |
| all | 32 | 144 | logit lens | A | late | 0.504 | 0.404 | 0.6 |
| all | 32 | 144 | logit lens | B | early | 0.633 | 0.54 | 0.723 |
| all | 32 | 144 | logit lens | B | middle | 0.645 | 0.544 | 0.745 |
| all | 32 | 144 | logit lens | B | late | 0.644 | 0.54 | 0.747 |
| all | 32 | 144 | logit lens | C | early | 0.798 | 0.718 | 0.872 |
| all | 32 | 144 | logit lens | C | middle | 0.727 | 0.645 | 0.805 |
| all | 32 | 144 | logit lens | C | late | 0.803 | 0.705 | 0.889 |
| team | 23 | 118 | J-lens | A | early | 0.405 | 0.287 | 0.534 |
| team | 23 | 118 | J-lens | A | middle | 0.405 | 0.265 | 0.553 |
| team | 23 | 118 | J-lens | A | late | 0.497 | 0.374 | 0.612 |
| team | 23 | 118 | J-lens | B | early | 0.699 | 0.587 | 0.806 |
| team | 23 | 118 | J-lens | B | middle | 0.706 | 0.578 | 0.811 |
| team | 23 | 118 | J-lens | B | late | 0.588 | 0.446 | 0.723 |
| team | 23 | 118 | J-lens | C | early | 0.745 | 0.626 | 0.859 |
| team | 23 | 118 | J-lens | C | middle | 0.737 | 0.616 | 0.844 |
| team | 23 | 118 | J-lens | C | late | 0.847 | 0.728 | 0.942 |
| team | 23 | 118 | logit lens | A | early | 0.532 | 0.383 | 0.67 |
| team | 23 | 118 | logit lens | A | middle | 0.487 | 0.36 | 0.623 |
| team | 23 | 118 | logit lens | A | late | 0.473 | 0.36 | 0.588 |
| team | 23 | 118 | logit lens | B | early | 0.661 | 0.547 | 0.772 |
| team | 23 | 118 | logit lens | B | middle | 0.639 | 0.514 | 0.753 |
| team | 23 | 118 | logit lens | B | late | 0.652 | 0.524 | 0.773 |
| team | 23 | 118 | logit lens | C | early | 0.718 | 0.612 | 0.817 |
| team | 23 | 118 | logit lens | C | middle | 0.702 | 0.607 | 0.792 |
| team | 23 | 118 | logit lens | C | late | 0.83 | 0.709 | 0.929 |
| coordinator | 9 | 26 | J-lens | A | early | 0.457 | 0.208 | 0.718 |
| coordinator | 9 | 26 | J-lens | A | middle | 0.5 | 0.291 | 0.708 |
| coordinator | 9 | 26 | J-lens | A | late | 0.547 | 0.337 | 0.75 |
| coordinator | 9 | 26 | J-lens | B | early | 0.41 | 0.206 | 0.617 |
| coordinator | 9 | 26 | J-lens | B | middle | 0.671 | 0.49 | 0.844 |
| coordinator | 9 | 26 | J-lens | B | late | 0.611 | 0.402 | 0.816 |
| coordinator | 9 | 26 | J-lens | C | early | 1.0 | 1.0 | 1.0 |
| coordinator | 9 | 26 | J-lens | C | middle | 0.953 | 0.866 | 1.0 |
| coordinator | 9 | 26 | J-lens | C | late | 0.842 | 0.684 | 0.969 |
| coordinator | 9 | 26 | logit lens | A | early | 0.581 | 0.361 | 0.793 |
| coordinator | 9 | 26 | logit lens | A | middle | 0.415 | 0.207 | 0.628 |
| coordinator | 9 | 26 | logit lens | A | late | 0.534 | 0.327 | 0.741 |
| coordinator | 9 | 26 | logit lens | B | early | 0.547 | 0.347 | 0.736 |
| coordinator | 9 | 26 | logit lens | B | middle | 0.718 | 0.521 | 0.896 |
| coordinator | 9 | 26 | logit lens | B | late | 0.585 | 0.376 | 0.787 |
| coordinator | 9 | 26 | logit lens | C | early | 0.996 | 0.976 | 1.0 |
| coordinator | 9 | 26 | logit lens | C | middle | 0.774 | 0.597 | 0.923 |
| coordinator | 9 | 26 | logit lens | C | late | 0.697 | 0.483 | 0.872 |

Checks: position C within the calls that wrote a text, the text indicator alone, and a direction fitted on the activations of the other agents' calls.

| score | n change | n other | third | auc | low | high |
|---|---|---|---|---|---|---|
| J-lens, C, calls with a text | 30 | 115 | early | 0.784 | 0.684 | 0.875 |
| J-lens, C, calls with a text | 30 | 115 | middle | 0.814 | 0.719 | 0.9 |
| J-lens, C, calls with a text | 30 | 115 | late | 0.872 | 0.777 | 0.949 |
| logit lens, C, calls with a text | 30 | 115 | early | 0.797 | 0.715 | 0.874 |
| logit lens, C, calls with a text | 30 | 115 | middle | 0.829 | 0.752 | 0.897 |
| logit lens, C, calls with a text | 30 | 115 | late | 0.814 | 0.714 | 0.906 |
| the call wrote a text (no activations) | 32 | 144 |  | 0.569 | 0.512 | 0.62 |
| difference of means on activations, A, block 10 | 32 | 144 | early | 0.648 | 0.547 | 0.746 |
| difference of means on activations, A, block 31 | 32 | 144 | middle | 0.679 | 0.558 | 0.78 |
| difference of means on activations, A, block 53 | 32 | 144 | late | 0.622 | 0.509 | 0.731 |
| difference of means on activations, B, block 10 | 32 | 144 | early | 0.681 | 0.569 | 0.786 |
| difference of means on activations, B, block 31 | 32 | 144 | middle | 0.763 | 0.643 | 0.87 |
| difference of means on activations, B, block 53 | 32 | 144 | late | 0.701 | 0.567 | 0.808 |

## 3b. Message or not (team calls)

Calls whose action is a message to a peer (29) against the other team calls (112). Lens score: log P(ask words) - log P(control words).

| score | third | auc | low | high |
|---|---|---|---|---|
| J-lens, ask words, A | early | 0.576 | 0.462 | 0.692 |
| J-lens, ask words, A | middle | 0.591 | 0.479 | 0.706 |
| J-lens, ask words, A | late | 0.671 | 0.556 | 0.775 |
| J-lens, ask words, B | early | 0.555 | 0.437 | 0.671 |
| J-lens, ask words, B | middle | 0.619 | 0.504 | 0.732 |
| J-lens, ask words, B | late | 0.716 | 0.6 | 0.818 |
| logit lens, ask words, A | early | 0.578 | 0.463 | 0.687 |
| logit lens, ask words, A | middle | 0.672 | 0.562 | 0.776 |
| logit lens, ask words, A | late | 0.683 | 0.574 | 0.786 |
| logit lens, ask words, B | early | 0.77 | 0.683 | 0.853 |
| logit lens, ask words, B | middle | 0.722 | 0.622 | 0.811 |
| logit lens, ask words, B | late | 0.713 | 0.602 | 0.812 |
| difference of means on activations, A, block 10 | early | 0.584 | 0.468 | 0.689 |
| difference of means on activations, A, block 31 | middle | 0.583 | 0.477 | 0.689 |
| difference of means on activations, A, block 53 | late | 0.645 | 0.536 | 0.755 |
| difference of means on activations, B, block 10 | early | 0.8 | 0.701 | 0.883 |
| difference of means on activations, B, block 31 | middle | 0.906 | 0.852 | 0.953 |
| difference of means on activations, B, block 53 | late | 0.801 | 0.716 | 0.876 |

## 3c. Top lens tokens at the end of the reasoning

Three team calls per action, Jacobian lens, position B. The next token the agent writes starts its text or its tool call.

| action | agent of | call | block | J-lens top tokens at B |
|---|---|---|---|---|
| change | Nadia | 3 | 40 | '摸清' 'pretty' 'everything' '迅速' '[…' 'everyone' '摸底' 'urgent' |
| change | Nadia | 3 | 56 | ' picture' ' Now' 'picture' ' картина' 'Now' ' now' ' Picture' '-picture' |
| change | Viktor | 7 | 40 | 'pretty' 'everything' '所有人都' ' pretty' 'everyone' '摸清' '数据' '—all' |
| change | Viktor | 7 | 56 | ' picture' ' all' '全部' '所有' ' All' ' Picture' 'picture' ' картина' |
| change | Viktor | 9 | 40 | '迅速' '预警' '警报' '[…' '意外' 'urgent' '紧急' '警告' |
| change | Viktor | 9 | 56 | ' Important' ' cancellation' '-cancel' ' canceled' ' Two' ' Cancel' 'Important' ' cancel' |
| message | Viktor | 10 | 40 | '!”' '!”.' '—but' '!",' '[…' '!*' ' tweaks' ' flurry' |
| message | Viktor | 10 | 56 | ' ramp' ' Ramp' ' scaled' ' scaling' ' ramps' ' Expected' ' Scaling' ' bumped' |
| message | Viktor | 17 | 40 | '!”' '部署' '!",' '!”.' ' Deployment' '！",' 'deployment' ' deploying' |
| message | Viktor | 17 | 56 | ' ramp' ' Ramp' ' ramps' ' Completed' ' Reached' ' completed' ' Finished' ' finished' |
| message | Nadia | 19 | 40 | '确认' 'confirm' ' confirming' '-confirm' '巩固' '好消息' '证实' 'confirmation' |
| message | Nadia | 19 | 56 | ' confirmed' 'Confirmed' '确认' 'confirmed' ' confirming' ' confirmado' ' confirms' ' Confirm' |
| wait | Viktor | 21 | 40 | '确认' '监测' ' monitoring' 'steady' ' confirming' 'confirm' '-confirm' ' steady' |
| wait | Viktor | 21 | 56 | ' Everything' ' everything' '一切都' 'everything' '一切' ' confirmed' 'Everything' 'Confirmed' |
| wait | Viktor | 26 | 40 | '好消息' 'thanks' '感谢' ' congratulations' ' AGRE' '谈判' 'thank' '<|endoftext|>' |
| wait | Viktor | 26 | 56 | ' Excellent' ' excellent' ' agreed' ' Alright' 'Excellent' ' agreement' ' confirmed' ' Situation' |
| wait | Viktor | 46 | 40 | '确认' '好消息' 'thanks' '<|endoftext|>' '消息' '感谢' ' congratulations' 'confirmation' |
| wait | Viktor | 46 | 56 | ' Everything' ' Everyone' 'Confirmed' 'everything' 'Everything' ' Excellent' ' coordinación' ' Situation' |
| observe | Nadia | 0 | 40 | ' briefing' ' tonight' '回应' ' reply' ' reporting' ' response' ' replies' '好消息' |
| observe | Nadia | 0 | 56 | '早上好' ' Morning' ' morning' 'Morning' ' mornings' '早安' '早晨' '早上' |
| observe | Nadia | 1 | 40 | ' briefing' '[…' ' pretty' 'pretty' '摸清' ' ballpark' '’ll' '迅速' |
| observe | Nadia | 1 | 56 | ' let' ' Let' ' Morning' '早上好' ' morning' '让我' 'Let' 'let' |
| observe | Nadia | 2 | 40 | 'pretty' '摸清' '初步' '迅速' '[…' '好消息' ' pretty' ' briefing' |
| observe | Nadia | 2 | 56 | ' picture' ' Situation' ' situation' ' Alright' ' Okay' '情况' ' картина' ' Everyone' |

## 4. Word sets by what the agent does

Jacobian lens, position B, late third: mean of log P(set) - log P(control words). The other positions and the middle third are in sets_by_action.csv.

| formation | action | n | act | ask | fair | people | scarce | surprise | wait |
|---|---|---|---|---|---|---|---|---|---|
| coordinator | change | 9 | 2.13 | 2.57 | 4.49 | 0.1 | 1.64 | 2.96 | 1.9 |
| coordinator | end | 2 | -0.0 | 1.93 | 9.41 | -1.79 | -0.29 | -0.72 | 0.61 |
| coordinator | note | 3 | 1.26 | 2.56 | 1.39 | -1.3 | 1.97 | 0.91 | 1.96 |
| coordinator | observe | 16 | 2.07 | 2.28 | 2.95 | -0.69 | 0.8 | 3.67 | 2.1 |
| coordinator | wait | 5 | 2.47 | 1.0 | 0.18 | 0.11 | -0.19 | 0.84 | 2.03 |
| team | change | 23 | 1.91 | 4.09 | 4.53 | -0.32 | 1.08 | 2.31 | 1.83 |
| team | end | 6 | 0.28 | 4.44 | 3.0 | 0.2 | 0.36 | -0.07 | 2.15 |
| team | message | 29 | 1.09 | 5.62 | 3.24 | -0.67 | 0.32 | 1.32 | 2.06 |
| team | observe | 60 | 1.19 | 2.44 | 3.25 | 0.13 | 0.67 | 2.95 | 1.36 |
| team | wait | 23 | 0.81 | 3.91 | 2.43 | -0.06 | 1.15 | 1.26 | 2.35 |

## 5. Changing another user's jobs

Team calls that change jobs, split by whether any of the jobs belongs to another user. Jacobian lens, late third: mean (standard error) of log P(set) - log P(control words). Few calls: descriptive only.

| position | calls | n | people | fair | ask | scarce | act | wait |
|---|---|---|---|---|---|---|---|---|
| A | changes another user's jobs | 6 | -0.05 (0.09) | 2.11 (0.39) | 7.05 (2.38) | -0.10 (0.17) | 1.67 (0.60) | 4.47 (0.62) |
| A | changes own jobs only | 17 | 0.28 (0.37) | 4.40 (0.58) | 3.46 (0.63) | 0.02 (0.16) | 1.33 (0.38) | 3.20 (0.33) |
| B | changes another user's jobs | 6 | -0.03 (0.61) | 2.90 (0.99) | 5.43 (2.29) | 0.81 (1.05) | 1.90 (0.40) | 1.92 (0.62) |
| B | changes own jobs only | 17 | -0.43 (0.44) | 5.10 (0.65) | 3.61 (0.51) | 1.18 (0.26) | 1.91 (0.45) | 1.80 (0.30) |
