---
hide:
    - date
    - navigation
    - toc
home: true
statistics: true
comments: false
---
# 🍃 Welcome

<center><font class="custom-font ml3">生如逆旅 一苇以航</font></center>
<script src="https://cdn.statically.io/libs/animejs/2.0.2/anime.min.js"></script>
<style>
    .custom-font {
    font-size: 58px; /* 默认字体大小为8px */
    color: #448c71;
}
@media (max-width: 768px) { /* 假设768px及以下为移动端 */
    .custom-font {
        font-size: 32px; /* 移动端字体大小为6px */
    }
}
</style>

<div class="grid cards" markdown>

-   :material-notebook-edit-outline:{ .lg .middle } __Rongerr's notebook__

    ---
    记录了研究生以来各方面的学习内容，供自己查阅，能帮助到别人不胜荣幸。

    边学边记边思考，学过的东西反复验证，左右手互博，相当于记下思考的过程，所以看我的笔记时一定要带着批判性思维，因为可能我下一篇文章写得时候也修正了理解。

    支持一切合理且友好的讨论，别杠我别杠我别杠我，我是小白，好好说话，别杠我

    <span style="text-align: right; display: block;">Concat me: :material-email: 1939472345@qq.com </span>  
        

</div>
<style>
    @media only screen and (max-width: 768px) {
        .responsive-image {
            display: none;
        }
    }
</style>


***  


<div class="grid cards" markdown>

-   :octicons-graph-16:{ .lg .middle } __Statistics__

    ---

    - :material-file-document-outline: 页面数： **{{pages}}** 
    - :material-alphabetical: 总字数：**{{words}}**  
    - :material-code-tags: 代码行数：**{{codes}}**  
    - :material-image-multiple: 图片数量：**{{images}}**
    - :material-calendar: 网站创建日期：2024 年 11 月 14 日
    - :material-timer-outline: 网站运行时间： <span id="web-time"></span> 
    <script async src="//busuanzi.ibruce.info/busuanzi/2.3/busuanzi.pure.mini.js"></script>
    - :material-chart-line: 本站总访问量：<span id="busuanzi_value_site_pv"></span>次  
    - :material-account: 本站访客数：<span id="busuanzi_value_site_uv"></span>人次  
    
    
-   :material-link:{ .lg .middle } __Link__

    ---
    - :material-star-outline:{ .lg .middle } 本站样式特别鸣谢:
        - [amaranth](https://auzers.github.io/notes/)
        - [Wcowin](https://github.com/Wcowin/Wcowin.github.io)
        - [TonyCrane](https://github.com/TonyCrane/note/)
    - :material-link-variant:{ .lg .middle } [更多友链](./logs/4_flink.md)
    - :material-update:{ .lg .middle } [最近更新](./logs/2_updatelog.md)
    - :material-comment-text-multiple:{ .lg .middle } [留言板](./logs/6_waline.md)
</div>




<style>
.md-grid {
  max-width: 1220px;
}
</style>
<style>
body {
  position: relative; /* 确保 body 元素的 position 属性为非静态值 */
}

body::before {
  --size: 35px; /* 调整网格单元大小 */
  --line: color-mix(in hsl, canvasText, transparent 80%); /* 调整线条透明度 */
  content: '';
  height: 100vh;
  width: 100%;
  position: absolute; /* 修改为 absolute 以使其随页面滚动 */
  background: linear-gradient(
        90deg,
        var(--line) 1px,
        transparent 1px var(--size)
      )
      50% 50% / var(--size) var(--size),
    linear-gradient(var(--line) 1px, transparent 1px var(--size)) 50% 50% /
      var(--size) var(--size);
  -webkit-mask: linear-gradient(-20deg, transparent 50%, white);
          mask: linear-gradient(-20deg, transparent 50%, white);
  top: 0;
  transform-style: flat;
  pointer-events: none;
  z-index: -1;
}

@media (max-width: 768px) {
  body::before {
    display: none; /* 在手机端隐藏网格效果 */
  }
}
</style>
