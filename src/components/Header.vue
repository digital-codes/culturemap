<template>
    <header v-if="route === 'chat'" class="header">
        <h2>Mapping Culture in Karlsruhe</h2>
    </header>
</template>

<style scoped>
.header {
    text-align: center;
    padding: 2rem;
}

@media screen and (max-width: 600px) {
    .header {
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