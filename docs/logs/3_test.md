# 功能测试页面

<div id="box1" style="width: 50px; height: 50px; background: blue; position: absolute;"></div>
<div id="box2" style="width: 50px; height: 50px; background: green; position: absolute; top: 60px;"></div>

<script src="https://cdn.statically.io/libs/animejs/2.0.2/anime.min.js"></script>
<script>
  var timeline = anime.timeline({
    easing: 'easeInOutQuad',
    duration: 1000
  });

  timeline
    .add({
      targets: '#box1',
      translateX: 250
    })
    .add({
      targets: '#box2',
      translateX: 250,
      offset: '-=500' // 第二个动画提前 500ms 开始
    });
</script>