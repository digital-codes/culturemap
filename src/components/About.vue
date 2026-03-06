<script setup lang="ts">
</script>

<template>
    <section class="about" aria-labelledby="about-heading">
        <h2 id="about-heading" class="visually-hidden">{{ $t('message.aboutHeading') }}</h2>
        <p>{{ $t('message.about') }}</p>
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
</style>
