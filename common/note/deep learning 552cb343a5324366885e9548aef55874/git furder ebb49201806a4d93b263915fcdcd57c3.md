# git furder

# Issue tracker

게시판 같은 역할 할 수 있음

# Gerrit

일종의 투표소

어떤 코드를 반영하는 것이 좋을지 상호 평가할 수 있다

# patch

opensource pull 권한이 없는사람이 자신의 아이디어를 파일 제작자에게 줄 수 있는 좋은 방법

# Pull request(Merge request)

다른사람들에게 내가 작업한 코드를 검토해달라고 하는 요청

# git merge tool

conflict 가 발생했을때 수동으로 코드를 merge 하기 위한 tool

# git workflow

branch 를 효과적으로 사용하기위해 다른사람들이 정해놓은 규칙들

예를들면 branch의 이름을 다음과 같이 약속하고 사용하는 것이다

master : 최종본

develop : 실험본(실험할 때마다 새로운 branch 를 만들어서 좋으면 master 에 통합시킴)

※git flow 도 참고

# cherry-pick

main branch(보통 master 이라 함)와 다른 branch 를 통째로 합치는게 아닌 다른 branch 의 일부분만을 merch 하고 싶을 때 사용함

# rebase

merge 를 하고 timeline 을 좀더 깔끔하게 만드는 기술

![Untitled](git%20furder%20ebb49201806a4d93b263915fcdcd57c3/Untitled.png)

기존 방식의 timeline

![Untitled](git%20furder%20ebb49201806a4d93b263915fcdcd57c3/Untitled%201.png)

rebase