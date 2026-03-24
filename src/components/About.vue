<script setup lang="ts">
</script>

<template>
    <section class="about" aria-labelledby="about-heading">
        <h2 id="about-heading" class="visually-hidden">{{ $t('message.aboutHeading') }}</h2>
        <p>{{ $t('message.about') }}</p>
        <video controls class="video">
            <source src="/video/cards.webm" type="video/webm">
            Your browser does not support the video tag.
        </video>
    </section>
</template>

<style scoped>
.visually-hidden {
    position: absolute;
    width: 1px;
    height: 1px;
    margin: -1px;
    padding: 0;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
}
.video {
    height: 50vh;
    width: auto;
    margin-top: 1rem;
}

@media (max-width: 600px) {
    .video {
        height: 40vh;
    }
}

</style>
