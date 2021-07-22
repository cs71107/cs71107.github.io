---
title: "BOJ 22306 트리의 색깔과 쿼리 2"
date: 2021-07-22 21:00:00
categories:
- PS
tags:
- [tree, online query]
---

[BOJ 22306](https://www.acmicpc.net/problem/22306)

위 링크에서 문제 지문을 확인할 수 있다.

[트리의 색깔과 쿼리](https://www.acmicpc.net/problem/17469) 문제완 달리, **온라인**으로 쿼리들을 처리해야 하는게 핵심이다.

이 글은 다음 개념들에 대한 지식을 독자가 이미 가지고 있다고 가정하고 서술한다.

- offline query
- tree의 euler tour traversal
- tree의 dfs ordering
- segment tree(index-tree)
- small to large

원본 문제에선 offline으로 해결이 가능해 small to large를 적당히 활용하면 쉽게 풀 수 있다.
하지만 온라인으로 어떻게 할 지가 어렵다.

여러 가지 방식이 있을 것이고, 실제로 검수진 및 출제자의 풀이도 다양하다.

여기에선 내가 사용한 $O(nlog^{2}n+q)$풀이를 소개한다.

1번 쿼리를 어떻게 해결할지 생각해보자.

우선 오프라인일 때는 small to large를 역순으로 합쳐가면서 진행했으니, 지금은 역으로 small to large를 정점 집합을 분리시키면서 할 수 있지 않을까?라는 생각에서 시작한다.

그런 다음과 같은 두 가지 어려운 점이 있다.

1) 두 개로 쪼개지는 집합 중, 어느 집합이 더 작은 쪽인가?
2) 두 개로 나눠진 집합에서, 서로 다른 색깔의 개수를 어떻게 관리할 것인가?

우선 1)을 해결한다. 현재 어떤 정점 $a$에 대해서 그 부모 정점에 대해 끊는 상황이라 가정하자.

현재 $a$가 속해 있는 집합에 대해서, 그 정점들의 집합을 tree의 root에서 dfs 했을 때 나타나는 순서대로 정렬한 것을 $\lbrace v_{0}, v_{1}, \dots, v_{k-1} \rbrace$라고 하자. 그렇다면, $a$의 subtree에서 $a$와 같은 집합에 속해 있는 정점들의 indices는 **구간**을 이룬다. 즉, 어떤 $0 \leq i \leq j \leq k-1$에 대해 $i \sim j$이 $a$의 subtree에 속한 정점들의 indices가 된다. 이 사실에 착안하면, 풀이의 실마리를 찾을 수 있다.

이제 위의 $i,j$를 찾는 다면, $j-i+1$이 $a$의 subtree 중에서 현재 집합에 있는 정점들의 개수가 된다. 그러므로, $2(j-i+1) \geq k$라면, $a$의 subtree가 아니면서 현재 집합에 있는 정점들을 순회하면서 업데이트를 진행하고, $2(j-i+1) < k$이면 $a$의 subtree이면서 현재 집합에 속해 있는 정점들을 순회하면서 업데이트를 진행시키면 된다.

이를 효율적인 시간내에 구현할 수 있게 해주는 자료구조로는 set,map 등의 bbst 들과 segment tree가 있다.

2)의 경우, 이제 집합들을 분리할 수 있으므로, 각 집합에 대해 서로 다른 색깔이 얼마나 있는지 관리하고, 각각의 개수를 역시 배열이나 set등으로 관리하고, 업데이트의 경우 분리한 두 집합 중 작은 쪽의 정점을 돌며 업데이트 해주면 해결할 수 있다. 처음엔 서로 다른 색깔의 개수를 저장하고, 업데이트를 진행하며 어떤 색깔의 개수가 0이 되는 시점에 집합 내의 서로 다른 색깔의 개수를 하나 줄여주면 된다.

서로 다른 색깔의 개수는 정점의 개수를 넘을 수 없으므로, 정점, 색깔 모두 $O(nlogn)$에 비례하게 공간을 차지하게 만들 구현할 수 있다.

2번 쿼리는 이제 각 집합마다 색깔의 개수를 관리해주고 있으니, $O(1)$에 답변할 수 있다.

구현이 기므로, 주요 함수들과 배열에 대한 설명한다.

함수:

- dfs() : 최초에 한번 호출되는 dfs 함수. tree를 순회하면서 dfs ordering 상으로 몇 번째인지 등을 정한다.
- init() : 필요한 초기화 작업을 한다. 최초에 모든 정점들이 포함되어 있는 집합에 대한 업데이트 진행.
- getsum() : 위에서 언급한 $j-i+1$값을 구해주는 역할
- getid() : 주어진 범위에서 현재 집합에 남아 있는 정점이 있다면, 그것의 id, 즉 집합이 처음 생성 됐을 때, 그 집합 내에서 dfs ordering 순서가 몇 번째였는지 반환한다. 그렇지 않다면 INF를 return.
- update_er() : segment tree에서 현재 지우고자 하는 정점에 관련된 정보를 업데이트.
- update_add() : 집합을 두 개로 쪼갰을 때, 작은 쪽 집합의 정점에 대한 정보를 업데이트
- update_set() : 1번 쿼리가 들어왔을 때, 새로운 집합을 만들고, 기존 집합에 대한 업데이트와 새로운 집합에 대한 초기화 작업을 진행한다.

배열:

- id : 각 집합 내 정점의 dfs ordering에서의 순서를 **순서대로** 저장한다.
- col: 각 집합이 최초로 생성됐을 때, 그 때의 서로 다른 색깔을 **색깔 번호의 순서대로** 저장한다.
- cal: 각 집합의 색깔에 대해, 그 색깔이 현재 집합 내에 얼마나 있는지 그 개수를 저장한다.


자세한 구현은 다음 코드를 참고하면 된다. 시간 복잡도는 상기한 대로 $O(nlog^{2}n+q)$이다.

```cpp
#include <bits/stdc++.h>
#define f first
#define s second
#define MOD 1000000007
#define PMOD 998244353
#define pb(x) push_back(x)
using namespace std;

typedef long long int ll;
typedef pair<int,int> pii;
typedef pair<ll,ll> plii;
typedef pair<int, pii> piii;
const int INF = 1e9+10;
const ll LINF = 1LL*INF*INF;
const int MAXN = 1e5+10;
const int MAXM = 5e3+10;

priority_queue<int> pq;
vector<vector<int> > graph;
queue<int> que;

int par[MAXN];
int A[MAXN];

int in[MAXN];
int out[MAXN];
int ver[MAXN];

int bb[MAXN];

vector<int> id[MAXN];
vector<int> col[MAXN];

int cid[MAXN];

vector<int> cal[MAXN];
vector<int> tree[MAXN];
vector<int> caltree[MAXN];

int ccal[MAXN];

int seq = 0;

int snum = 0;

void dfs(int here){

    int there;

    in[here] = seq;
    seq++;

    for(int i=0;i<graph[here].size();i++){
        there = graph[here][i];
        dfs(there);
    }

    out[here] = seq-1;
    return;
}

void init(int n){

    graph = vector<vector<int> > (n+1);

    for(int i=2;i<=n;i++){
        graph[par[i]].push_back(i);
    }

    dfs(1);

    for(int i=1;i<=n;i++){
        ver[in[i]] = i;
    }

    for(int i=1;i<=n;i++){
        col[0].push_back(A[i]);
    }

    sort(col[0].begin(),col[0].end());
    col[0].erase(unique(col[0].begin(),col[0].end()),col[0].end());

    int csz = (int)col[0].size();

    ccal[0] = csz;

    cal[0] = vector<int> (csz);

    int cidx = -1;

    for(int i=1;i<=n;i++){
        cidx = lower_bound(col[0].begin(),col[0].end(),A[i])-col[0].begin();
        cal[0][cidx]++;
    }

    int base = 1;

    for(;base<n;base<<=1);

    bb[0] = base;

    tree[0] = vector<int> (base<<1,INF);
    caltree[0] = vector<int> (base<<1);

    for(int i=0;i<n;i++){
        tree[0][base+i] = i;
        caltree[0][base+i] = 1;
    }

    for(int i=base-1;i>=1;i--){
        tree[0][i] = tree[0][(i<<1)];
        caltree[0][i] = caltree[0][(i<<1)]+caltree[0][(i<<1)|1];
    }

    id[0] = vector<int> (n);

    for(int i=0;i<n;i++){
        id[0][i] = i;
    }

    return;
}

inline int getsum(int curid,int L,int R){

    int res = 0;

    while(L<=R){

        if(L&1){
            res += caltree[curid][L];
            L++;
        }
        if(!(R&1)){
            res += caltree[curid][R];
            R--;
        }
        L>>=1; R>>=1;
    }

    return res;
}

inline int getid(int curid,int L,int R){

    int res = INF;
    int rres = INF;

    while(L<=R){

        if(L&1){
            if(!(res^INF))res = tree[curid][L];
            L++;
        }
        if(!(R&1)){
            if(tree[curid][R]^INF)rres = tree[curid][R];
            R--;
        }
        L>>=1; R>>=1;
    }

    if(res^INF)return res;
    else return rres;
}

inline void update_er(int curid,int tmp){

    tree[curid][tmp] = INF;
    caltree[curid][tmp] = 0;
    tmp>>=1;

    while(tmp){

        tree[curid][tmp] = min(tree[curid][(tmp<<1)],tree[curid][(tmp<<1)|1]);
        caltree[curid][tmp] = caltree[curid][(tmp<<1)]+caltree[curid][(tmp<<1)|1];
        tmp>>=1;

    }

    return;
}

inline void update_add(int curid,int nxtid,int base,int idx){

    int curin = id[curid][idx];

    int curver = ver[curin];
    int curcol = A[curver];

    update_er(curid,base+idx);

    id[nxtid].push_back(curin);
    col[nxtid].push_back(curcol);

    return;
}

void update_set(int a){

    int curid = cid[a];

    int Lid = lower_bound(id[curid].begin(),id[curid].end(),in[a])-id[curid].begin();
    int Rid = upper_bound(id[curid].begin(),id[curid].end(),out[a])-id[curid].begin();

    Rid--;

    int base = bb[curid];

    int curtot = caltree[curid][1];

    int cursub = getsum(curid,base+Lid,base+Rid);

    snum++;
    int nxtid = snum;

    int idx = -1;

    if((cursub<<1)>=curtot){

        while(true){
            idx = getid(curid,base,base+Lid-1);

            if(idx^INF){
                update_add(curid,nxtid,base,idx);
            }
            else {
                break;
            }
        }

        while(true){
            idx = getid(curid,base+Rid+1,(base<<1)-1);

            if(idx^INF){
                update_add(curid,nxtid,base,idx);
            }
            else {
                break;
            }
        }
    }
    else {

        while(true){
            idx = getid(curid,base+Lid,base+Rid);

            if(idx^INF){
                update_add(curid,nxtid,base,idx);
            }
            else {
                break;
            }
        }

    }

    sort(col[nxtid].begin(),col[nxtid].end());
    col[nxtid].erase(unique(col[nxtid].begin(),col[nxtid].end()),col[nxtid].end());

    int nxtsz = (int)id[nxtid].size();
    int ncsz = (int)col[nxtid].size();

    int nxt_base = 1;

    for(;nxt_base<nxtsz;nxt_base<<=1);

    tree[nxtid] = vector<int> ((nxt_base<<1),INF);
    caltree[nxtid] = vector<int> ((nxt_base<<1));

    cal[nxtid] = vector<int> (ncsz);

    for(int i=0;i<nxtsz;i++){
        tree[nxtid][nxt_base+i] = i;
        caltree[nxtid][nxt_base+i] = 1;
    }

    for(int i=nxt_base-1;i>=1;i--){
        tree[nxtid][i] = tree[nxtid][(i<<1)];
        caltree[nxtid][i] = caltree[nxtid][(i<<1)]+caltree[nxtid][(i<<1)|1];
    }

    bb[nxtid] = nxt_base;

    int ncidx = -1;
    int nxtcol = 0;
    int nxtver = 0;

    for(int i=0;i<nxtsz;i++){
        nxtver = ver[id[nxtid][i]];
        nxtcol = A[nxtver];
        ncidx = lower_bound(col[nxtid].begin(),col[nxtid].end(),nxtcol)-col[nxtid].begin();

        cid[nxtver] = nxtid;

        cal[nxtid][ncidx]++;
    }

    ccal[nxtid] = ncsz;

    int ccidx = -1;

    for(int i=0;i<ncsz;i++){
        ccidx = lower_bound(col[curid].begin(),col[curid].end(),col[nxtid][i])-col[curid].begin();

        cal[curid][ccidx]-=cal[nxtid][i];
        if(!cal[curid][ccidx]){
            ccal[curid]--;
        }
    }

    return;
}

int main()
{
    int n,m,k,a,b,x,y,q;
    int sum = 0;
    int cnt = 0;
    int mx = 0;
    int mn = INF;
    int cur = 0, idx = -1;
    int tc;
    int ty;

    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    cin>>n>>q;

    for(int i=2;i<=n;i++){
        cin>>par[i];
    }

    for(int i=1;i<=n;i++){
        cin>>A[i];
    }

    init(n);

    int res = 0;

    for(int i=1;i<n+q;i++){
        cin>>ty>>k;

        a = k^res;

        if(ty&1){
            update_set(a);
        }
        else {
            res = ccal[cid[a]];
            cout<<res<<"\n";
        }
    }

    return 0;
}

```
