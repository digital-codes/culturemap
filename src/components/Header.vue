<template>
    <header class="header">
        <h1 v-if="route === 'chat'">Mapping Culture in Karlsruhe</h1>
    </header>
</template>

<style scoped>
.header {
    margin-top: 3rem;
    text-align: center;
    padding: 2rem;
}

@media screen and (max-width: 600px) {
    .header {
        margin-top: 2rem;
        padding: 1rem;
    }
}

</style>    

<script setup lang="ts">
defineProps({
    route: {
        type: String,
        required: true
    }
})

</script>