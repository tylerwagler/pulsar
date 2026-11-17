# The official-continuation scorer.  The repo root's Makefile owns the rule (one object list, one set of flags);
# `make -C tools/quality-testing` builds it from here.
ROOT := ../..

score_official:
	$(MAKE) -C $(ROOT) tools/quality-testing/score_official

clean:
	rm -f score_official score_official.o score_official.d
.PHONY: score_official clean
