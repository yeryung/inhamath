# docker command

![Untitled](docker%20command%20df755cd8f71e4a56b9fe635489a51f08/Untitled.png)

reference

[https://www.youtube.com/watch?v=EbTJtanJUfE&list=PLuHgQVnccGMDeMJsGq2O-55Ymtx0IdKWf&index=3](https://www.youtube.com/watch?v=EbTJtanJUfE&list=PLuHgQVnccGMDeMJsGq2O-55Ymtx0IdKWf&index=3)

# image

```bash
docker pull [image_name]
```

docker hub 에서 image_name 을 설치

```bash
docker images
```

설치된 image 를 보여줌

```bash
docker run {option} [image_name] {command}
```

image_name 을 실행시킴

```bash
docker run --name [container_name] [image_name]
```

container_name 을 이름으로 갖는 image_name 이 생성된다

```bash
docker rmi [image_name]
```

지움

# container

```bash
docker ps
```

만들어진 container 를 보고싶을 때

```bash
docker ps -a
```

정지된 container 까지 보여줌

```bash
docker stop [container_name/id]
```

container_name 을 가진 container 를 정지함

```bash
docker start [container_name/id]
```

정지되었던 container_name 을 가진 container 를 재시작함

```bash
docker logs [container_name/id]
```

로그 출력줌

```bash
docker logs -f [container_name/id]
```

로그를 실시간으로 보여줌 

```bash
docker rm [container_name/id]
```

지움