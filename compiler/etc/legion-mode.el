; Copyright 2014 Stanford University and Los Alamos National Security, LLC
;
; Licensed under the Apache License, Version 2.0 (the "License");
; you may not use this file except in compliance with the License.
; You may obtain a copy of the License at
;
;     http://www.apache.org/licenses/LICENSE-2.0
;
; Unless required by applicable law or agreed to in writing, software
; distributed under the License is distributed on an "AS IS" BASIS,
; WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
; See the License for the specific language governing permissions and
; limitations under the License.
;

(require 'generic-x)

(define-generic-mode 'legion-mode
  '("//" ("/*" . "*/"))
  '("array" "as" "assert" "bool" "color" "coloring"
    "double" "downregion" "else" "false" "float"
    "for" "if" "import" "in" "int" "int8" "int16"
    "int32" "int64" "isnull" "ispace" "let" "new"
    "null" "pack" "partition" "read" "reads"
    "reduce" "reduces" "region" "return" "struct"
    "task" "true" "uint" "uint8" "uint16" "uint32"
    "uint64" "unpack" "upregion" "using" "var"
    "void" "while" "write" "writes" "zip")
  '(("task\\s-+\\([A-Za-z][A-Za-z0-9_]*\\)" 1 'font-lock-function-name-face)
    ("struct\\s-+\\([A-Za-z][A-Za-z0-9_]*\\)" 1 'font-lock-type-face)
    ("let\\s-+\\([A-Za-z][A-Za-z0-9_]*\\)" 1 'font-lock-variable-name-face)
    ("var\\s-+\\([A-Za-z][A-Za-z0-9_]*\\)" 1 'font-lock-variable-name-face)
    ("for\\s-+\\([A-Za-z][A-Za-z0-9_]*\\)" 1 'font-lock-variable-name-face)
    ("\\([A-Za-z][A-Za-z0-9_]*\\)\\s-*:" 1 'font-lock-variable-name-face))
  '("\\.lg$")
  nil
  "A mode for Legion files"
)

(provide 'legion-mode)
