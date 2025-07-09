Place a compressed Chromium archive named ``chromium.tar.xz`` in this directory.
The bootstrap scripts will extract it to ``resources/chromium/bin`` when no
system Chromium install is detected. After adding the archive you can execute
the pipeline offline via:

```bash
MSK_REMOTE_URL="https://nbia.cancerimagingarchive.net/viewer/?study=..." \
MSK_AUTH_TOKEN="eyJhbGciOiJI..." \
python bootstrap_pipeline.py
```
