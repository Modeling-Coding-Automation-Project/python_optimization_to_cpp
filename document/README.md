# python_optimization_to_cpp ドキュメント

本ドキュメントは、`optimization_utility` と `python_optimization` の使い方を、`sample` 内の実コードに基づいて整理したものです。  
アルゴリズム内部ではなく、**利用者がどう組み立てて使うか**に焦点を当てています。

## 目次

1. [全体像と利用フロー](./01_overview_and_workflow.md)
2. [Pythonサンプル実装ガイド](./02_python_sample_guide.md)
3. [C++サンプル実装ガイドとPython/C++インターフェース対応](./03_cpp_sample_and_interface_mapping.md)
4. [コード生成機能ガイド（Deploy）](./04_code_generation_guide.md)

## このドキュメントの前提

- リポジトリ内の以下を対象に説明しています。
  - `optimization_utility`
  - `python_optimization`
  - `sample`
- 外部ライブラリは本リポジトリの機能を支えるために利用されていますが、説明はこの `document` フォルダ内で完結するよう再構成しています。

