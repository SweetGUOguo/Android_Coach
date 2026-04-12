FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
# FROM nvcr.io/nvidia/pytorch:24.05-py3

RUN apt-get update && apt-get install -y \
    # cmake \ 
    # environment-modules \
    git \
    # libgtest-dev \
    sudo \
    vim \
    libnss3 \
    # libnss3-dev \
    libxkbfile1 \
    # libpulse-dev \
    libdrm2 \
    # libxi-dev \
    libx11-6 \
    libxi6 \
    pulseaudio \
    curl \
    # qemu-kvm \
    # libvirt-daemon-system \
    # libvirt-clients bridge-utils \
    wget \
    unzip \
    python3-pip \
    cpu-checker \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && npm i --location=global appium \
    && appium driver install uiautomator2 \
    && pip install Appium-Python-Client \
    # && apt-get clean autoclean \
    # && /bin/rm -f /core.*
    # && adduser `id -un` kvm \
    # && chown -R $USER:kvm /dev/kvm
    && mkdir /app

WORKDIR /app
RUN wget https://builds.openlogic.com/downloadJDK/openlogic-openjdk/8u412-b08/openlogic-openjdk-8u412-b08-linux-x64.tar.gz \
    && tar -xvf ./openlogic-openjdk-8u412-b08-linux-x64.tar.gz \
    && mv ./openlogic-openjdk-8u412-b08-linux-x64 "/app/.jdk" \
    && wget https://dl.google.com/android/repository/sdk-tools-linux-4333796.zip \
    && unzip sdk-tools-linux-4333796.zip -d "/app/.android"

ENV JAVA_HOME="/app/.jdk/"
ENV PATH="$JAVA_HOME/bin:$PATH"
ENV ANDROID_HOME="/app/.android/"
ENV SDK="$ANDROID_HOME"
ENV ANDROID_SDK_ROOT="$ANDROID_HOME"
ENV PATH="$SDK/emulator:$SDK/tools:$SDK/tools/bin:$SDK/platform-tools:$PATH"

RUN yes | sdkmanager "platform-tools" "platforms;android-28" "platforms;android-33" "emulator" \
    && yes | sdkmanager "system-images;android-28;google_apis;x86_64" "system-images;android-33;google_apis;x86_64"\
    && yes | sdkmanager "build-tools;28.0.0" "build-tools;33.0.0" \
    && wget --no-check-certificate "https://drive.usercontent.google.com/download?id=1QhBNhvZTQhJzugUsDv-rRrevm8qsb03U&export=download&authuser=0&confirm=t&uuid=a768ee32-7ba8-4d16-a095-4379e0571c97&at=AEz70l6BxwdF6eqfQ62CzCS4kejq:1741664257701" -O test_Android.zip \
    && mkdir "$ANDROID_HOME/avd" \
    && unzip test_Android.zip -d "$ANDROID_HOME/avd" \
    && wget --no-check-certificate "https://drive.usercontent.google.com/download?id=1SJ79gdO7whgUod3HnuS87aOKihRk1i-U&export=download&authuser=0&confirm=t" -O docker-file.zip \
    && unzip docker-file.zip -d "/app" \
    && unzip "/app/docker-file/skins.zip" -d "$ANDROID_HOME" \
    && mv "/app/docker-file/Pixel_7_Pro_API_33.ini" "$ANDROID_HOME/avd" \
    && unzip "/app/docker-file/Pixel_7_Pro_API_33.avd.zip" -d "$ANDROID_HOME/avd" \
    && mkdir -p "$ANDROID_HOME/system-images/android-33/google_apis/" \
    && yes | unzip "/app/docker-file/x86_64.zip" -d "$ANDROID_HOME/system-images/android-33/google_apis/" \
    && rm -rf "/app/docker-file/" \
    && rm -rf docker-file.zip \
    && sed -i 's|skin.path = [^"]*|skin.path = /app/.android/skins/pixel_7_pro|' /app/.android/avd/Pixel_7_Pro_API_33.avd/config.ini

ENV ANDROID_AVD_HOME="$ANDROID_HOME/avd"

RUN wget --no-check-certificate "https://drive.usercontent.google.com/download?id=19FnCaEh6VzR99Za-pvUoR-60iNe8mgom&export=download&authuser=0&confirm=t" -O android-eval-avd.zip \
    && mkdir "$ANDROID_HOME/tmp" \
    && unzip android-eval-avd.zip -d "$ANDROID_HOME/tmp" \
    && rm -rf android-eval-avd.zip \
    && chmod -R 027 "$ANDROID_HOME"


RUN cd /app/.android/avd \
    && mv Pixel_7_Pro_API_33.avd Pixel_7_Pro_API_33.avd.ori \
    && cp -r /app/.android/tmp/Pixel_7_Pro_API_33.avd ./ \
    && cp Pixel_7_Pro_API_33.avd.ori/config.ini Pixel_7_Pro_API_33.avd/ \
    && cp Pixel_7_Pro_API_33.avd.ori/hardware-qemu.ini Pixel_7_Pro_API_33.avd/



COPY requirements.txt .
RUN chmod -R 027 "$ANDROID_HOME" \
    && chmod -R 027 "$JAVA_HOME" \
    && pip install -r requirements.txt 
    # && pip install --upgrade flash-attn --no-build-isolation
    # net-tools
    # && apt-get install -y libxkbfile1
    # libnss3 libnss3-dev libxkbfile1 
    # libpulse-dev libdrm2 libxi-dev
    # libxi-dev
    # && chmod 555 /dev/kvm

# uninstall uvloop
RUN pip uninstall -y uvloop

# Add sudo support
RUN echo "%users ALL = (ALL) NOPASSWD: ALL" >> /etc/sudoers
