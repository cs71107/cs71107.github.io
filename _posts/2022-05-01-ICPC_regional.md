---
title: "ICPC 2021 본선 후기"
date: 2022-05-01 14:00:00
categories:
- contest
tags:
- ICPC
---

## 들어가기 전에 ##

ICPC 예선결과가 나오고 나서, 일단은 각자 할일이 있었기 때문에, 각자 할일을 하면서 ICPC 본선을 준비하게 됐다.

## 대회 준비 과정 ##

ICPC 예선 연습처럼, 본선 역시 csi2021이라는 계정을 통해서 연습을 했다.

연습 장소는 ICPC 본선을 칠 스터디 카페에서 해보면 제일 좋겠지만, 시간과 비용의 문제로 그냥 서울대 윗공대 slab의 방 하나와, 컴퓨터공학부 과방에서 연습을 진행했다.

첫번째 연습의 경우 2020 Asia Yokohama Regional, 두번째 연습의 경우 Latin America Regional Contests 2019을 사용해서 연습을 진행했다.

각 연습의 결과를 첨부한다.

2020 Asia Yokohama Regional:

![](/image/2020_Yokohama_practice.png)

Latin America Regional Contests 2019:

![](/image/2019_Latin_practice.png)

2020 Asia Yokohama의 경우 LCM of GCDs란 문제가 예전에 Sait님에 의해 풀려 있었는데, 내가 그걸 체크하지 않은 상태로 연습에 돌입했었다. 단, Sait신도 명확하게 어떤 문제였는진 당시에 기억하지 못했다.

Latin 2019의 경우 Sait신이 연습 중에 라이브로 DB를 박아서 다이아를 푸는 것을 목격했다. Sait신....

보다 시피 연습 결과가 상당히 좋았다. 두 연습 모두 2개 정도의 문제를 제외하면 모두 풀었고, 하위 다이아 문제도 푸는데 성공했음을 볼 수 있다. 그래서 연습 과정에서 어느 정도 자신감을 얻은 상태로 ICPC 본선에 들어갔다.

본선을 친 장소는 예선과 똑같은 서울대 입구 주변의 [토즈 모임센터](https://map.naver.com/v5/entry/place/1005741752?c=14132112.5605060,4506402.3293062,13,0,0,0,dh&placePath=%2Fhome%3Fentry=pll)에서 진행하기로 결정했다. 프린터기는 여전히 Sait님의 운반하시기로 했다.

대회 전에 ICPC 본선용 키트 (스웨트, 현수막 등등)이 도착했고, 난 ICPC 스웨트를 입고 서울대 입구에 나갔다. 거기에서 팀원들과 합류한 이후, 먼저 스터디 카페에 짐을 풀었다. 그리고 근처에 KFC가 있어 간단하게 요기를 했고, 현수막, 노트북, 프린터기 등 대회 준비를 위한 기본 환경 설정을 마쳤다. 대회 시작 전에, 긴장을 풀기 위해서 잡담을 좀 나눴는데, 우리끼리는 한 5위 정도 하면 우린 성공인 것 같다고 얘기를 했다. 아래 부터는 실제 대회 내 진행과정이다.


## 대회 진행 과정 ##

12:00~ - 대회시작, 일단 문제지를 뽑았고, 평소대로 내가 ABCD, Sait님이 EFGH, IHHI가 JIKL을 맡았고, 문제들을 읽기 시작했다.

12:10~20 - 우선 A가 어려운 트리 문제(시간을 들여야 하는 타입)이라는 걸 깨닫고 넘겼고, B를 보니 전형적인 문제라서, 짤 수 있겠다는 생각이 들어서 빠르게 구현해서 바로 AC를 맞았다. B 솔브.

12:24~ - 그 사이에 IHHI가 C를 풀어서 구현, 바로 AC를 맞았다. 그때 난 D를 읽고 있었다. D를 읽으니 풀 수 있겠다는 감이 왔고, 구현을 시작했다. C 솔브.

12:44 - 열심히 짰으나.... 결과는 WA. 그 순간 나는 멘붕이 왔고, 일단 프린트를 해서 디버깅을 하기로 결정했다.

12:58 - 내가 열심히 디버깅을 하고 있을 때, F의 풀이를 알아낸 Sait신이 빠르게 구현을 해서 AC를 받았다. 그리고 난 내 구현의 방향성이 살짝 잘못됐다는 걸 깨닫고, Sait신이 AC를 받자마자 바로 구현을 시작했다. F 솔브.

13:03~13:09 - D를 제출했으나, 또 틀렸다... 다시 멘붕이 왔고, 우리팀의 페널티는 나 혼자 만들어내고 있는 상황이었다. 다시 출력 후 디버깅에 들어갔다. 그리고 정말 어이 없는 구현 실수로 WA를 받았음을 깨달았다. 빠르게 고치고 다시 냈고, 마침내 AC! D 솔브.

13:10~41 - 그사이에 IHHI가 L의 풀이를 알아냈고, 열심히 구현을 하기 시작했다. IHHI가 열심히 구현한 L을 제출했고... 결과는 AC! L 솔브.

13:41~58 - IHHI가 L을 구현하고, 난 다른 문제를 읽고 있는 동안에, Sait신이 K가 SCPC 때 나왔던 바로 그 유형이라면서 L의 결과가 나오자마자 열심히 코딩을 하기 시작했다. 팀노트에 있는 KMP구현 등등을 살짝 뒤지고 구현을 마친 후, 제출했더니 결과는 AC. K 솔브.

13:58~14:12 - Sait신이 K를 열심히 구현할 때, 나와 IHHI는 G의 풀이를 의논하고 있었다. 내가 G와 비슷한 문제를 코포에서 만났던 기억이 있기 때문에, 왠지 dfs+dp일 것 같다라는 의견을 말했고, IHHI도 풀이를 확신해서 바로 구현에 들어갔다. 난 그때 아주 강한 확신은 없었지만, 어쨌거나 IHHI는 구현을 마쳤고, 바로 AC를 맞았다. G 솔브.

14:12~14:34 - 우리가 G에 신경 쓰고 있을 동안, 우리는 E가 수학이라는 것을 파악하고 Sait신에게 토스를 한 상황이었다. Sait신이 뭔가를 열심히 뚝딱뚝딱하고, 우리에게 의견을 구했다. 우리가 맞는 것 같다고 하자 Sait신이 E 구현에 들어갔고, AC를 받았다. E 솔브.

이 시점에서 스코어보드를 확인했고, 현재 전체 2위, 서울대 1위임을 확인했다.

![](/image/2021_ICPC_Regional_mid_scoreboard.png)

스코어 보드를 확인한 결과, 문제가 4개 남았기 때문에 각자 한 문제씩 잡고 남은 시간을 보내는 것으로 했다. H가 많이 어려워 보였기 때문에, 일단 H를 제외하고, 트리+자구스런 문제인 A를 내가, I를 IHHI가, J를 Sait님이 이렇게 나누었다.

14:34~16:09 - A의 풀이를 열심히 고민한 결과, HLD+세그먼트 트리를 쓰면 문제를 풀 수 있다는 결론을 도출했다. 단, 풀이에서 쓰는 주요 알고리즘만 봐도 느껴지듯이, 적어도 나에겐 굉장한 구현량을 요구하는 문제였다. 그렇게 때문에 일단 다른 문제 중 풀이가 나온 것이 있으면 바로 교체할 수 있도록 하되, 일단 A 구현을 시작했다. A는 굵직한 풀이도 구현이 많이 필요하지만, 디테일한 처리도 필요한 문제였다. 그렇게 1차적으로 구현을 마치고 나자, 400줄이 넘는 구현이 나왔고... 디버깅도 힘들겠다 판단해서 예제가 나오는 걸 보자 그냥 바로 냈다. 결과는 WA.... 

16:09~ - 이제 시간이 1시간도 남지 않아서, A가 WA가 뜨자 모두 A 디버깅에 달려 들었다. 하지만 내 코드가 워낙 길고 + 나만 쓰는 특징들이 있던터라 같이 디버깅하기가 힘들었다. 그 뒤로 내가 잘못 구현한 것들을 하나씩 찾아냈고, 그때마다 제출해보았으나 WA 2번을 더 쌓는... 결과만 낳았다.

16:52~ - 최후의 수단으로 작은 입력을 몇 개 만들어서 잘못된 답을 내는 입력이 있는지 찾아보기 시작했고, 운이 좋게도 해당 입력을 얼마 지나지 않아 찾을 수 있었다. 그리고 내가 정말 바보 같은 실수를 했다는 걸 깨달았다. 해당하는 오류를 빠르게 고쳤고, 지금까지 만들었던 입력 + 예제들을 모두 넣어본 후, 제대로 결과가 나오는 것을 확인했고, 제출한 뒤... AC. 결국 종료 8분 전에야 A를 맞을 수 있었다. 그 뒤 남은 문제는 I,J가 있었는데 사실상 풀이를 찾지 못해서 이대로 대회를 마쳤다.

## 대회 결과 ##

대회가 끝나기전에 캡처한, 프리즈가 풀리기 전의 스코어보드는 다음과 같다.

![](/image/2021_ICPC_Regional_freeze.png)

다른 팀들이 얼마나 풀었는지 모르겠지만, 꽤 높은 순위일 것이라 예상한 상태로 뒷정리를 마치고, 가까운 고깃집으로 갔다.

고깃집에서 식사를 하면서, 대회 결과 방송을 봤다. 프리즈가 풀리고 나서, 우리 순위는 다음과 같았다.

![](/image/2021_ICPC_Regional_final_scoreboard.png)


## 총평 ##

전체 5등. 은상으로 대회를 마무리하게 됐다. 충분히 만족스런 성적이었다. 말이 씨가 된다고, 대회 전에 얘기했던 그 등수가 그대로 실제 등수가 됐다. J를 풀었더라면이라는 감정이 없다면 거짓말이지만, 우리 팀은 우리 팀이 할 수 있는 최선을 다했다고 생각한다.

본대회에서 난 우리 팀이 제출한 AC외의 제출을 혼자서 제출하며(...), 민폐가 될 뻔했지만, 다행히 A를 풀어서 어느 정도 상쇄를 할 수 있게 됐다.

대회 문제 풀이의 경우에는 내가 너무 늦게 글을 써서(..) 기억이 잘 나지 않고, 나중에 기회가 된다면 업소링을 하면서 업데이트를 하던가 할 예정이다. 그 외에 [구사과님의 글](https://koosaga.com/284)에 풀이가 잘 나와 있다.

대회 본 시점과 이 글을 쓰는 시점이 약 6개월(..)간 차이가 있다. 너무 늦은 감이 있지만, 나와 함께 ICPC를 치른 CSI팀원 모두, 친절하게 ICPC 등록을 도와준 눕장 yclock, 마지막으로 매일 다이아 정도의 어려운 문제를 푸는 것을 요구하면서 실력을 향상시키는데 도움을 준 분들 모두에게 이 글을 통해 감사를 전한다.
