# Stacked-AutoEncoder-Model

27127

### exp1

visualization_exp1.bin
```
self.ae1 = LDAutoEncoderLayer(input_size, 6000) # 27127
self.ae2 = LDAutoEncoderLayer(6000, 3000)
self.ae3 = LDAutoEncoderLayer(3000, 1500)
self.ae4 = LDAutoEncoderLayer(1500, 750)
self.ae5 = LDAutoEncoderLayer(750, 375)
self.ae6 = LDAutoEncoderLayer(375, output_size) # 3
```

visualization_exp1.bin
```
self.ae1 = LDAutoEncoderLayer(input_size, 512) # 27127
self.ae2 = LDAutoEncoderLayer(512, 256)
self.ae3 = LDAutoEncoderLayer(256, output_size) # 3
```

### References

https://github.com/Dierme/latent-gan