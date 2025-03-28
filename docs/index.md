---
statistics: true
---
# 首页

<center>
<div id="rcorners">
  <script src="https://sdk.jinrishici.com/v2/browser/jinrishici.js" charset="utf-8"></script>
  <div id="poem_sentence"></div>
  <div id="poem_info"></div>
  <script type="text/javascript">
  jinrishici.load(function(result) {
      var sentence = document.querySelector("#poem_sentence")
      var info = document.querySelector("#poem_info")
      sentence.innerHTML = result.data.content
      info.innerHTML =  '《' + result.data.origin.title + '》' + result.data.origin.author + '【' + result.data.origin.dynasty + '】'
  });
  </script>
</div> 
</center>

<center>
  <!-- 引入 Google Fonts 中的 "Long Cang" 字体 -->
  <link href="https://fonts.googleapis.com/css2?family=Long+Cang&display=swap" rel="stylesheet">

  <!-- 自定义样式 -->
  <style>
    #poem_sentence, #poem_info {
      font-family: "Long Cang", sans-serif; /* 设置字体为 Long Cang */
      font-size: 20px; /* 调整字体大小 */
      line-height: 1.5; /* 设置行距，增强可读性 */
      text-align: center; /* 居中对齐 */
    }
  </style>

  <!-- 今日诗词的功能代码 -->
  <script src="https://sdk.jinrishici.com/v2/browser/jinrishici.js" charset="utf-8"></script>
  <div id="poem_sentence"></div>
  <div id="poem_info"></div>
  <script type="text/javascript">
    jinrishici.load(function (result) {
      var sentence = document.querySelector("#poem_sentence");
      var info = document.querySelector("#poem_info");
      sentence.innerHTML = result.data.content;
      info.innerHTML = "《" + result.data.origin.title + "》" + result.data.origin.author + " · " + result.data.origin.dynasty;
    });
  </script>
</center>

## 简介

: Rongerr's notebook。

记录了研究生以来各方面的学习内容，供自己查阅的同时希望能帮助到更多的人。

## 统计

本网站共有 {{pages}} 个页面，{{words}} 字，{{codes}} 行代码，{{images}} 张图片。

## 致谢

本网站的建设使用或参考了以下内容：

- [Material for MkDocs](https://github.com/squidfunk/mkdocs-material)
- [鹤翔万里的笔记本](https://github.com/TonyCrane/note/)
- [Mkdocs-Wcowin 博客主题](https://github.com/Wcowin/Mkdocs-Wcowin)
- [giscus](https://github.com/giscus/giscus)
- [数学家是我理想](https://space.bilibili.com/181990557)
- [王木头学科学](https://space.bilibili.com/504715181?spm_id_from=333.337.0.0)
- [wmathor](https://wmathor.com/index.php/category/Deep-Learning/)
- [Just for Life](https://muyuuuu.github.io)
- [大白话AI](https://space.bilibili.com/9045161)
- [Rayman小何同学|VAE](https://www.bilibili.com/video/BV1Ax4y1v7CY?spm_id_from=333.788.videopod.sections&vd_source=ddd7d236ab3e9b123c4086c415f4939e)
- [deep_thoughts](https://www.bilibili.com/video/BV1sG411s7vV/?spm_id_from=333.337.search-card.all.click&vd_source=ddd7d236ab3e9b123c4086c415f4939e)
- [RethinkFun](https://www.bilibili.com/video/BV1dtSuY7Evj?spm_id_from=333.788.player.switch&vd_source=ddd7d236ab3e9b123c4086c415f4939e)
- 

