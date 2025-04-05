# mkdocs 常用命令

[https://squidfunk.github.io/mkdocs-material/reference/admonitions/](https://squidfunk.github.io/mkdocs-material/reference/admonitions/)

## 折叠框

### 文本折叠框

```html
<details>
<summary>说明</summary>
<p>
你想折叠的一大段内容
</p>
</details>
```

<details>
<summary>内容概要</summary>
<p>
你想折叠的一大段内容
</p>
</details>

### 代码折叠框

~~~markdown
<details>
<summary>说明：</summary>
<p>
```python
```
</p>
</details>
~~~

效果：

<details>
<summary>说明：</summary>
<p>
```python
```
</p>
</details>

<details>
<summary>Old mkdocs.yml</summary>
<p>
```yaml
```
</p>
</details>

## 描述框

```markdown
!!! note
    This is a note.
```

!!! note
    This is a note.
    

```markdown
??? question "What is the meaning of life, the universe, and everything?"
```

??? question "What is the meaning of life, the universe, and everything?"

```markdown
!!! abstract
    you can use 
    - note
    - abstract
    - info
    - tip
    - success
    - question
    - warning
    - failure
    - danger
    - bug
    - example
    - quote
```

```markdown
!!! note "note 标题"

    Lorem 
```

!!! note "note 标题"

    Lorem 

```
??? note "折叠 note"

    Lorem 
```

??? note "折叠 note"

    Lorem 


```
???+ note "默认展开 note"

    Lorem 
```

???+ note "默认展开 note"

    Lorem 