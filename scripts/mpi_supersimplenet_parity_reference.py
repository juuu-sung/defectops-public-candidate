"""Independent adaptor/head and loss transcription, not a second Anomalib call.

Adapted from Blaž Rolih's SuperSimpleNet ICPR implementation, commit
9c6867c3e1ef36d9e9fa7722a7dfdc8f9e41e301, model/supersimplenet.py
and common/loss.py. Backbone features and additive synthetic noise are injected
identically for algebra parity; this is NOT a full paper reproduction.

MIT License
Copyright (c) 2024 Blaž Rolih
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def build_reference(channels=1536, hidden=1024, classification_channels=128):
    import torch
    from torch import nn

    class Adapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Conv2d(channels, channels, 1)

        def forward(self, x):
            return self.projection(x)

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.seg_head = nn.Sequential(nn.Conv2d(channels, hidden, 1),
                nn.BatchNorm2d(hidden), nn.LeakyReLU(.2, inplace=True),
                nn.Conv2d(hidden, 1, 1, bias=False))
            self.cls_conv = nn.Sequential(nn.Conv2d(channels + 1, classification_channels, 5, padding="same"),
                nn.BatchNorm2d(classification_channels), nn.ReLU(inplace=True))
            self.cls_fc = nn.Linear(classification_channels * 2 + 2, 1)

        def forward(self, x):
            f = torch.nn.functional
            maps = self.seg_head(x)
            cls = self.cls_conv(torch.cat((x, maps.detach()), dim=1))
            pooled = torch.cat((f.adaptive_max_pool2d(cls, 1), f.adaptive_avg_pool2d(cls, 1),
                               f.adaptive_max_pool2d(maps, 1).detach(),
                               f.adaptive_avg_pool2d(maps, 1).detach()), dim=1).flatten(1)
            return maps, self.cls_fc(pooled).flatten()

    class Reference(nn.Module):
        def __init__(self):
            super().__init__()
            self.adaptor = Adapter()
            self.segdec = Head()

        def forward(self, features, additive_noise):
            adapted = self.adaptor(features)
            return self.segdec(torch.cat((adapted, adapted)) + additive_noise)

    return Reference()


def reference_loss(maps, scores, masks, labels):
    """Author's probability-BCE focal, versus Anomalib's logits-BCE focal.

Do not silently replace this with SSNLoss: saturation differences are evidence.
"""
    import torch
    from torch.nn import functional as f

    def focal(logits, target):
        probability = logits.sigmoid().float()
        target = target.float()
        pt = probability * target + (1 - probability) * (1 - target)
        return (f.binary_cross_entropy(probability, target, reduction="none") * (1 - pt)**4).mean()

    normal = torch.clamp(maps[masks == 0] + .5, min=0)
    anomaly = torch.clamp(-maps[masks > 0] + .5, min=0)
    zero = maps.new_zeros(())
    parts = {"map_focal": focal(maps, masks),
             "map_truncated_l1": (normal.mean() if normal.numel() else zero) +
                                 (anomaly.mean() if anomaly.numel() else zero),
             "classification_focal": focal(scores, labels)}
    return sum(parts.values()), parts
