;;; Directory Local Variables
;;; For more information see (info "(emacs) Directory Variables")

((nil . ((conda-project-env-path . "sry-env"))))
((python-mode . ((flycheck-disabled-checkers . (python-mypy python-pylint)))))
(with-eval-after-load 'lsp-mode
  (add-to-list 'lsp-file-watch-ignored-directories "[/\\\\]\\results\\"))

;; ((nil . ()))
